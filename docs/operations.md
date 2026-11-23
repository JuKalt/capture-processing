# Validate and resume

From the Capture Processing checkout, run
`./run.sh capture-validate --subject-root PATH` to check media, baseline SMPL-X and
metrics. Add `--require media smplx smplx_confidence mhr mhrfromsmplx metrics
semantics visualizations` for the full workflow, or specify only the stages
enabled for your capture. Add `--deep` to decode media and verify saved hashes
and semantic contents. The command returns nonzero for incomplete products.

Rerun the same processing command to resume. Full validation scans source and
processed pairs. Quick validation assumes source media is immutable and verifies
settings, calibration, fit and stage dependencies. Hold one writer per subject.
The lock file persists independently of ownership. A lock conflict exits with
code 75 so Device Jobs can retry it.

The standalone package namespace changes the tracked source code identity.
Before reusing caches produced by the combined repo, inspect them using
`./run.sh python -m capture_processing.tools.migrate_cache_signatures --output-root PATH`.
The tool reports by default. Apply only to a reviewable copy with known input
and model provenance until its report has been checked. Keep baseline MAMMA
intermediates, confidence outputs, semantic raw caches and the media receipts
for backfill.

For verification without licensed runtimes, use a development environment with
the `test` extra and run `python -m unittest discover -v`. The tests cover pipeline contracts and
recoverable failures. Model quality requires separate validation.
