"""Download and verify MAMMA assets without passing credentials on the command line."""

from getpass import getpass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen
from zipfile import ZipFile

from scripts.setup.setup_smplx_model import checksum, install

PUBLIC = {
    "data/weights/yolo/yolo12x.pt":
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x.pt",
    "data/weights/sam2/sam2.1_hiera_large.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "configs/sam2.1/sam2.1_hiera_l.yaml":
        "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
}
MAMMA = {
    "data/weights/ma_2d/mamma_mask_full_cvpr.ckpt": "weights/mamma_mask_full_cvpr.ckpt",
    "data/body_models/downsampled_verts/verts_512.pkl": "mamma_assets/verts_512.pkl",
}
SMPLX_ARCHIVE = "smplx_lockedhead_20230207.zip"
MAX_ATTEMPTS = 3


class CredentialsRejected(ValueError):
    """An upstream response indicates the gated download was denied."""


def credentials(account, registration, *, retry=False):
    user = None if retry else os.environ.get(f"{account}_USERNAME")
    password = None if retry else os.environ.get(f"{account}_PASSWORD")
    if not (user and password):
        if not sys.stdin.isatty():
            raise ValueError(
                f"{account} assets require an authorized account: {registration}. "
                f"Run interactively or provide {account}_USERNAME and {account}_PASSWORD."
            )
        print(f"Sign in with an authorized account from {registration}", flush=True)
        user = user or input(f"{account} username: ").strip()
        password = password or getpass(f"{account} password: ")
    if not user or not password:
        raise CredentialsRejected(f"No {account} credentials supplied")
    return urlencode(dict(username=user, password=password), quote_via=quote).encode()


def fetch_public(url, destination):
    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def fetch_mpi(url, destination, data):
    # Both upstream installers use Wget for the MPI gated POST endpoint.
    # Keep credentials out of argv and leave TLS verification enabled.
    with NamedTemporaryFile(dir=destination.parent, prefix=".mpi-login-") as form:
        form.write(data)
        form.flush()
        result = subprocess.run(
            ["wget", "--server-response", "--no-verbose", "--timeout=120",
             f"--post-file={form.name}", "--output-document", str(destination), url],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
        )
    if result.returncode:
        statuses = re.findall(rb"(?m)^\s*HTTP/\S+\s+(\d{3})\b", result.stderr)
        if statuses and int(statuses[-1]) >= 400:
            raise HTTPError(url, int(statuses[-1]), "Wget download failed", {}, None)
        reasons = {4: "network failure", 5: "TLS certificate verification failed"}
        reason = reasons.get(result.returncode, f"Wget exit code {result.returncode}")
        raise ValueError(f"MPI download failed for {destination.name}: {reason}")


def gated_url(domain, name):
    query = urlencode({"domain": domain, "resume": 1, "sfile": name}, safe="/")
    return f"https://download.is.tue.mpg.de/download.php?{query}"


def download(repo, assets, *, public=True, gated=True):
    def needed(name):
        path = repo / name
        return not path.is_file() or checksum(path) != assets[name]

    def save(source, name):
        install(source, repo / name, assets[name], label="MAMMA asset")

    def gated_downloads(account, registration, files):
        # Keep successful files if a later download is denied, and reuse one
        # login across all downloads from that provider.
        for attempt in range(MAX_ATTEMPTS):
            filename = files[0][1].name
            try:
                login = credentials(account, registration, retry=True) if attempt else credentials(account, registration)
                while files:
                    url, dest, consume = files[0]
                    filename = dest.name
                    try:
                        print(f"Downloading {filename} ...", flush=True)
                        fetch_mpi(url, dest, login)
                    except HTTPError as exc:
                        if exc.code not in (401, 403):
                            raise
                        exc.close()
                        raise CredentialsRejected(f"HTTP {exc.code}") from None
                    with dest.open("rb") as stream:
                        prefix = stream.read(256).lstrip().lower()
                    # The MPI download service also returns denial pages with HTTP 200.
                    if prefix.startswith((b"<!doctype html", b"<html")):
                        raise CredentialsRejected("the download server returned a web page instead of the file")
                    if b"error: file not found." in prefix:
                        raise CredentialsRejected("the download server reported the file was not found")
                    consume(dest)
                    files.pop(0)
                    dest.unlink()
                return
            except CredentialsRejected as exc:
                message = (f"{account} download rejected for {filename}: {exc}. "
                           f"Check download access and license acceptance at {registration}. "
                           "Signing into the website does not by itself verify access to this file.")
                if not sys.stdin.isatty() or attempt == MAX_ATTEMPTS - 1:
                    raise ValueError(message) from None
                print(f"{message} Try again ({MAX_ATTEMPTS - attempt - 1} attempts left).",
                      file=sys.stderr, flush=True)

    with TemporaryDirectory(prefix="vci-model-download-") as directory:
        scratch = Path(directory)
        if public:
            for name, url in PUBLIC.items():
                if name in assets and needed(name):
                    dest = scratch / Path(name).name
                    print(f"Downloading {dest.name} ...", flush=True)
                    fetch_public(url, dest)
                    save(dest, name)
                    dest.unlink()
        if not gated:
            return
        mamma_files = [
            (gated_url("mamma", remote), scratch / Path(name).name,
             lambda dest, name=name: save(dest, name))
            for name, remote in MAMMA.items() if name in assets and needed(name)
        ]
        missing_smplx = [name for name in assets if "/smplx_locked_head/" in name and needed(name)]
        if (mamma_files or missing_smplx) and not shutil.which("wget"):
            raise ValueError("GNU Wget is required to download MAMMA and SMPL-X assets; install wget and rerun setup.")
        if mamma_files:
            gated_downloads("MAMMA", "https://mamma.is.tue.mpg.de/", mamma_files)
        if missing_smplx:
            def unpack(archive):
                with ZipFile(archive) as package:
                    for name in missing_smplx:
                        leaf = Path(name).name
                        members = [member for member in package.namelist()
                                   if member.endswith(f"/smplx/{leaf}") or member == f"smplx/{leaf}"]
                        if len(members) != 1:
                            raise ValueError(f"Expected exactly one smplx/{leaf} in the upstream archive")
                        dest = scratch / leaf
                        with package.open(members[0]) as source, dest.open("wb") as output:
                            shutil.copyfileobj(source, output)
                        save(dest, name)

            gated_downloads("SMPLX", "https://smpl-x.is.tue.mpg.de/", [
                (gated_url("smplx", SMPLX_ARCHIVE), scratch / SMPLX_ARCHIVE, unpack)
            ])
