# Capture Processing examples

Run from the Capture Processing root after [setup](../docs/setup.md).
`./run.sh` runs the installed converter without shell activation. Replace the sample paths and IDs
with yours. The raw parent contains one folder per recording; each recording
has its own `calibration_dome.json` and `frame_*` directories. You can keep
another Conda environment active; the launcher uses the recorded converter
environment unless you explicitly override it.

## Process one recording first

From the Capture Processing checkout, check the authorized asset files before
launching GPU work. If your setup uses a custom MAMMA checkout, append
`--repo /path/to/mamma` to the asset check.

```bash
./run.sh python scripts/setup/setup_mamma_assets.py --check
```

```bash
./run.sh python scripts/datasets/process_vci_subject.py \
  /data/raw-captures example-subject /data/processed -- --workers 4

./run.sh capture-validate --subject-root /data/processed/example-subject --deep \
  --require media smplx smplx_confidence mhr mhrfromsmplx metrics semantics visualizations
```

The single-subject wrapper selects exactly `example-subject`. Arguments after
`--` go to `capture-process`. This example uses the default GPU stages.
The validator prints a JSON report with `"complete": true` on success and
exits nonzero if a required product is missing or invalid. This checks saved products without running inference.
Consult the [output schema](../docs/schema.md) for calibration, media and fit
paths and [validation and resume](../docs/operations.md) for incomplete stages.

## Process a batch with fewer stages

```bash
./run.sh capture-process /data/raw-captures /data/processed-baseline \
  --workers 4 --mhr-source none --skip-smplx-to-mhr \
  --no-semantic-parsing --no-mamma-confidence-fit

./run.sh capture-validate --subjects-root /data/processed-baseline --require media smplx metrics
```

This still requires the baseline MAMMA runtime and model assets. Use a separate
output root when comparing stage selections. Rerun the same command after
an interruption to resume. On immutable inputs, `--resume-validation quick`
reduces scanning; use the default validation when inputs may have changed.

## Reprocess part of a native dataset

List sequence, camera and frame IDs with the Multiview Data inspection example.
Then select a synchronized set of images and masks. This example starts from
native DNA data and runs only the baseline stages:

```bash
export DNA_RENDERING_SAMPLE_CODE=/data/dna/sample-code
./run.sh capture-reprocess dna /data/dna/Part1 /data/reprocessed-dna \
  --subject example-sequence --cameras 00 08 16 24 --frames 0 1 2 3 -- \
  --workers 4 --mhr-source none --skip-smplx-to-mhr \
  --no-semantic-parsing --no-mamma-confidence-fit

./run.sh capture-validate --subjects-root /data/reprocessed-dna --require media smplx metrics --deep
```

Keep the imported output separate from the source. The processed subject
manifest records the original frame and camera IDs. If calibration, masks,
or the required number of fitted views are missing, correct the selection
before running a larger import.

## Run with Device Jobs

Copy [processing-job.json](processing-job.json) and edit the raw data path in
`task.command`, `subjects`, the run and output roots, and each machine's checkout
and Python paths. The `python` field should point to the installed converter
environment, such as `/opt/miniforge/envs/vci-dataset/bin/python`. Set
`output_layout: "root"` so Capture Processing appends the subject ID once. The sample pins an explicit SAM3D interpreter on
the worker; edit it for your machine or remove the `--mhr-python` argument
and its value when using the default runtime from setup.

```bash
DEVICE_JOBS=/path/to/device-jobs/run.sh  # In the suite: "$PWD/../device-jobs/run.sh"
cp examples/processing-job.json /tmp/processing-job.json
# Edit /tmp/processing-job.json as described above.
"$DEVICE_JOBS" --config /tmp/processing-job.json --dry-run
"$DEVICE_JOBS" --config /tmp/processing-job.json --preflight-only
"$DEVICE_JOBS" --config /tmp/processing-job.json
```

Run the live command after reviewing the preview and passing preflight.
Validate the resulting
`output_root/<subject>` as in the single-recording example. Ensure runtime
tools such as `conda` are on `PATH` on each worker. See the [operations guide](../docs/operations.md) for cache migration.
