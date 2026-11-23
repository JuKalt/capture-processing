"""Verify gated download routing, integrity, and repeat runs without real credentials."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import parse_qs, urlsplit
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from zipfile import ZipFile

from scripts.setup import download_assets as downloader


class DownloadAssetsTests(unittest.TestCase):
    def setUp(self):
        # Download routing tests mock both clients, regardless of installed tools.
        wget = patch.object(downloader.shutil, "which", return_value="/usr/bin/wget")
        wget.start()
        self.addCleanup(wget.stop)

    def test_mpi_wget_uses_a_private_post_file_and_reports_denials(self):
        body = b"username=example&password=example%20with%2Bplus"
        url = downloader.gated_url("mamma", "weights/mamma_mask_full_cvpr.ckpt")

        def deny(args, **kwargs):
            form = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--post-file=")))
            self.assertEqual(form.read_bytes(), body)
            self.assertEqual(form.stat().st_mode & 0o777, 0o600)
            self.assertEqual(args[0], "wget")
            self.assertEqual(args[-1], url)
            self.assertNotIn("password", " ".join(args))
            self.assertNotIn("--no-check-certificate", args)
            return subprocess.CompletedProcess(args, 8, stderr=b"  HTTP/1.1 403 Forbidden\n")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader.subprocess, "run", side_effect=deny
        ):
            scratch = Path(directory)
            with self.assertRaises(HTTPError) as failure:
                downloader.fetch_mpi(url, scratch / "model", body)
            self.assertEqual(failure.exception.code, 403)
            self.assertEqual(list(scratch.iterdir()), [])

    def test_prompts_only_for_missing_groups_and_reuses_verified_files(self):
        names = list(downloader.PUBLIC) + list(downloader.MAMMA) + [
            f"data/body_models/smplx_locked_head/smplx/SMPLX_{kind}.npz"
            for kind in ("NEUTRAL", "FEMALE", "MALE")
        ]
        hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
        leaves = {Path(name).name: name for name in names}

        def fetch(url, destination, data=None):
            if destination.name == downloader.SMPLX_ARCHIVE:
                self.assertIn("domain=smplx", url)
                self.assertEqual(data, b"smplx-login")
                with ZipFile(destination, "w") as archive:
                    for name in names[-3:]:
                        archive.writestr(f"models_lockedhead/smplx/{Path(name).name}", name)
            else:
                if "domain=mamma" in url:
                    self.assertEqual(data, b"mamma-login")
                else:
                    self.assertIsNone(data)
                destination.write_text(leaves[destination.name])

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with patch.object(downloader, "fetch_public", side_effect=fetch) as request, patch.object(downloader, "fetch_mpi", side_effect=fetch) as mpi_request, patch.object(
                downloader, "credentials", side_effect=[b"mamma-login", b"smplx-login"]
            ) as prompt:
                downloader.download(repo, hashes)
                downloader.download(repo, hashes)
                self.assertEqual(prompt.call_count, 2)
                self.assertEqual(request.call_count, 3)
                self.assertEqual(mpi_request.call_count, 3)
            for name in names:
                self.assertEqual((repo / name).read_text(), name)

    def test_denied_mamma_download_reprompts_without_restarting_successful_files(self):
        names = list(downloader.MAMMA)
        hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
        responses = iter([names[0], "<html>Access denied</html>", names[1]])

        def fetch(url, destination, data=None):
            destination.write_text(next(responses))

        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader, "fetch_mpi", side_effect=fetch
        ) as requests, patch.object(downloader, "credentials", side_effect=[b"old", b"new"]) as prompt, patch.object(
            downloader.sys.stdin, "isatty", return_value=True
        ):
            downloader.download(Path(directory), hashes, public=False)
            self.assertEqual(prompt.call_args.kwargs, {"retry": True})
            self.assertEqual([request.args[2] for request in requests.call_args_list], [b"old", b"old", b"new"])
            for name in names:
                self.assertEqual((Path(directory) / name).read_text(), name)

    def test_smplx_denials_stop_after_three_tries_and_noninteractive_stops_after_one(self):
        name = "data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz"
        def deny(url, *_):
            self.assertEqual(parse_qs(urlsplit(url).query), {
                "domain": ["smplx"], "resume": ["1"], "sfile": [downloader.SMPLX_ARCHIVE],
            })
            raise HTTPError(url, 403, "Forbidden", {}, None)

        for interactive, count in ((True, 3), (False, 1)):
            with self.subTest(interactive=interactive), tempfile.TemporaryDirectory() as directory, patch.object(
                downloader, "fetch_mpi", side_effect=deny
            ), patch.object(downloader, "credentials", return_value=b"login") as prompt, patch.object(
                downloader.sys.stdin, "isatty", return_value=interactive
            ):
                with self.assertRaisesRegex(ValueError, "SMPLX download rejected"):
                    downloader.download(Path(directory), {name: "0" * 64}, public=False)
                self.assertEqual(prompt.call_count, count)

    def test_gated_checksum_failure_does_not_reprompt(self):
        name = next(iter(downloader.MAMMA))
        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader, "fetch_mpi", side_effect=lambda _, path, __: path.write_text("corrupt")
        ), patch.object(downloader, "credentials", return_value=b"login") as prompt, patch.object(
            downloader.sys.stdin, "isatty", return_value=True
        ):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                downloader.download(Path(directory), {name: "0" * 64}, public=False)
            self.assertEqual(prompt.call_count, 1)

    def test_invalid_public_response_never_becomes_an_installed_file(self):
        first = next(iter(downloader.PUBLIC))
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with patch.object(downloader, "fetch_public", side_effect=lambda _, path: path.write_text("error")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    downloader.download(repo, {first: "0" * 64}, gated=False)
            self.assertFalse((repo / first).exists())
