# VCI preprocessing data contract

[Conversion guide](../README.md) · [Operations](operations.md)

The portable subject manifest is `subject_manifest.json`, currently
`schema_version: 1`. Existing subjects without it are upgraded by rerunning their
normal preprocessing command. Paths in `products` are relative to the subject.
Cache manifests are operational state and may require validation after relocation.

## Input

The batch entry point expects a parent of recording directories. A recording has
`calibration_dome.json` (override with `--calibration`) and `frame_<integer>`
directories. Frame IDs must be unique and nonnegative; gaps retain their original
indices. A timestamp or frame rate is not inferred from frame numbering.

Each frame contains `image/color_<SOURCE_ID>` and matching masks under `mask/`,
`person_masks/`, or `masks/` (in that priority order). Images support `.avif`,
`.png`, `.jpg`, `.jpeg`; masks support
the extensions listed in `src/utils/transcode.py`. Source IDs are normalized to
uppercase and must be unique in calibration. Dropped source cameras are removed,
and retained cameras are sorted and renumbered contiguously as `cam00`, etc.
`camera_id_mapping.json` preserves processed-to-source IDs.

Calibration contains a nonempty `cameras` list. Each camera has `camera_id`,
`intrinsics.camera_matrix` (9 values), `intrinsics.resolution` (`[width,height]`),
and `extrinsics.view_matrix` (16 values). Flat and nested matrices are supported.
Transforms must be finite, rigid, and world-to-camera; focal lengths and integer
resolutions must be positive. Actual image and mask dimensions must match the
calibration. Mismatches are rejected before fitting.

World distances and translation are **meters**. Projection uses
`X_camera = R @ X_world + T`, then `K`. Inputs must already be undistorted and
oriented consistently with calibration; this converter does not undistort or
rotate them. Unit conventions are an input contract, not inferred from values.
Source metadata and calibration-repair provenance are retained in the manifest.

Missing image/mask pairs are omitted together and listed in
`missing_img_files.txt`. A missing source frame produces entries for all retained
cameras. Missing observations do not imply valid reconstructed poses; inspect
fit coverage and the manifest's source/fit frame IDs separately.

## Output

- `images/<camera>/color_<8-digit-frame>.jpg`: RGB JPEG.
- `masks/<camera>/mask_<8-digit-frame>.jpg`: grayscale JPEG. Binary 0/1 source
  foreground is normalized to 255; read JPEG masks with a threshold above 127.
  Masks are resized with nearest-neighbour interpolation. JPEG is lossy.
- `calibration.json`: each camera has flattened `K` (9), `R` (9), `T` (3).
  Resizing scales intrinsics to the exported image grid.
- `smpl_params.npz`: baseline SMPL-X. `global_orient` and `transl` are `[T,3]`,
  `body_pose` `[T,63]`, `jaw_pose` `[T,3]`, hands `[T,45]`, `expression` `[T,10]`,
  and shared `betas` `[1,B]`. Rotations are axis-angle in radians. New exports
  include integer `frame_ids`; legacy archives use row indices. The MAMMA export
  currently uses zero expression and does not export eye rotations. Hand poses
  are relative to a flat hand (build SMPL-X with `flat_hand_mean=True`); such
  archives contain `flat_hand_mean=True`. MAMMA fits relative to SMPL-X's mean
  hand, so the export adds that mean. Older MAMMA exports without the key are
  converted on the next pipeline run, without refitting.
- `smpl_params_confidence.npz`: the independent confidence-filtered variant.
- `mhr_params.npz`: SAM3D estimates with explicit `frame_ids`, view counts,
  reference cameras, checkpoint/completion state and diagnostics. Model fields
  retain SAM3D's parameter conventions; world projection needs its recorded
  reference camera, as described in the conversion guide.
- `mhr_from_smplx_params.npz`: official converted MHR. Contains `[T,204]`
  `lbs_model_params`, `[T,45]` `identity_coeffs`, `[T,72]` `face_expr_coeffs`,
  faces, frame IDs, per-frame fitting errors, units and conversion identity.
  Its world axes match SMPL-X; linear distances are **centimeters**.
- `fit_metrics.json`: status, coverage, residuals, shared silhouette selection,
  evaluation options, input signature and artifact identities. Unavailable
  measurements are null, with an explanation; they are not zero scores.
- `semantic_parsing/`: detailed/general labels, confidence, previews, schema,
  raw posteriors, metrics and cache state. Detailed labels use the 29-class
  Sapiens2 dome29 taxonomy; unknown foreground is 255. General labels use the
  fixed mapping in `schema.json`. Confidence is uint16 divided by 65535.
- `subject_manifest.json`: source metadata, known timing, IDs, coordinate
  assumptions, requested configuration, stage outcomes, products, fit coverage
  and provenance. Unknown FPS/timestamps/coverage remain null. A failed run
  records its error and cannot set manifest `complete` to true.
  `provenance.repository_dirty` is true when tracked code differed from
  `repository_revision` (null outside a git checkout). The standalone backfill
  tools (`fit_mamma_confidence`, `parse_subject_semantics`) mark their stages in
  the processing log and refresh stages, products, coverage and fit/semantic
  provenance of an existing manifest, appending to `provenance.backfills`; source,
  timeline and configuration still describe the last full pipeline run.

The processed-data adapter retains the common canonical keys and adds optional
`smplx_confidence`, `mhrfromsmplx`, `semantic_general`, `semantic_detailed`,
`semantic_confidence`, `semantic_schema`, `semantic_metrics`, `fit_metrics`, and
`subject_manifest` paths. Missing optional products are `None`.
Generic `labels` selects an existing legacy `labels/` first, then semantic general
labels. Select `semantic_detailed` explicitly for the detailed taxonomy.
Presence in the adapter is discovery; use the validator for completion evidence.

Automatic body fits and semantic labels require independent accuracy validation.
No unit test or pipeline completion flag certifies scientific annotation quality.
