# Set up processing

Use a Linux Git checkout with Conda on `PATH` and a compatible NVIDIA GPU and
driver for GPU stages. The installer creates the converter and isolated MAMMA
and Sapiens2 Conda environments, provisions Pixi for SAM3D Body and MHR, and
initializes nested submodules in the upstream checkouts. If the full VCI Suite
installer already set up this checkout, skip `./install.sh` and use its launcher.

```bash
./install.sh
./run.sh capture-process --help
```

The installer records the selected Conda environment (named `vci-dataset` by
default). `./run.sh` uses that record, so Conda need not be on `PATH` after
installation. Set `ENV_NAME` to select another installed environment, and
`VCI_CONDA_EXE` if its Conda executable is not found. No shell activation is
required. You can leave another Conda environment active during setup or
when running the launcher. Both run in a separate process without calling
the active environment’s deactivation hooks.

A sibling `multiview-data` checkout is used if present. Set
`MULTIVIEW_DATA_CHECKOUT` to choose a different checkout, or install from a
configured Python package index after Multiview Data is published.

The installer verifies existing files and downloads missing MAMMA and SMPL-X
assets from their original providers. It prompts for the two authorized
accounts if needed. Automatic downloads of gated MAMMA and SMPL-X assets require GNU Wget,
the client used by both upstream download scripts. Interactive sign-in allows up to three attempts per
provider. For noninteractive setup supply `MAMMA_USERNAME`,
`MAMMA_PASSWORD`, `SMPLX_USERNAME`, and `SMPLX_PASSWORD` through your runtime
credential store. To use an authorized local copy instead, set
`VCI_MAMMA_ASSET_SOURCE` to a directory laid out like the MAMMA checkout
(with `data/` and `configs/`). Use `--download-mamma-assets` to select upstream
downloads explicitly even when a local source is configured.

If the MAMMA site accepts your login but the installer still reports a web
page or a missing file, check whether the same account can download that
specific asset via the upstream MAMMA *Pipeline assets* panel. The download
server can return a web page for several causes, including missing file
access or temporary throttling. For an already authorized download, place
the files in the expected layout and pass `VCI_MAMMA_ASSET_SOURCE`; the
installer still verifies the pinned checksums.

When the SAM3D or Sapiens2 model files need downloading, setup can prompt for
a Hugging Face read token. Request access to both models before installing.
For unattended setup, supply an `HF_TOKEN` with the required access.

Use `./install.sh --skip-models` to build all environments without preparing
MAMMA, SAM3D or Sapiens2 models, or `./install.sh --converter-only` for the
converter without GPU stage runtimes. Individual scripts in `scripts/setup/`
can update just one runtime. Choose `MAMMA_CUDA_PROFILE=blackwell` for the
MAMMA CUDA 12.8 profile when your GPU requires it.

For raw captures, the default run enables MAMMA baseline and confidence fitting,
official SMPL-X-to-MHR conversion, SAM3D MHR, fit metrics and Sapiens2 labels.
Explicit overrides and stage flags are documented by `./run.sh capture-process --help`.
The converter detects the SAM3D Pixi interpreter in its checkout. Use
`--mhr-python` to override it.
External reprocessing additionally needs the native decoder dependencies of the
selected dataset reader.

A worker with a complete, authorized SAM3D model cache can run offline with
`HF_HUB_OFFLINE=1`. An unseeded worker still needs authenticated model access.
