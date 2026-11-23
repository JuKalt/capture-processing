"""Estimate multiview MHR parameters for one converted VCI subject."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import types
import warnings
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def _load_calibration(path: Path):
    import numpy as np

    data = json.loads(path.read_text())
    out = {}
    for cam_name, cam in data.items():
        K = np.asarray(cam["K"], dtype=np.float32).reshape(3, 3)
        R = np.asarray(cam["R"], dtype=np.float32).reshape(3, 3)
        T = np.asarray(cam["T"], dtype=np.float32).reshape(3)
        extr = np.eye(4, dtype=np.float32)
        extr[:3, :3] = R
        extr[:3, 3] = T
        out[cam_name] = {"K": K, "world_to_cam": extr, "P": K @ extr[:3, :]}
    return out


def _frame_ids(subject_root: Path, cam_names: list[str]) -> list[int]:
    if not cam_names:
        return []
    first = subject_root / "images" / cam_names[0]
    ids = []
    for path in sorted(first.glob("color_*.jpg")):
        stem = path.stem
        try:
            ids.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if ids:
        return ids
    for path in sorted(first.glob("*.jpg")):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return ids


def _frame_path(root: Path, cam_name: str, frame_id: int, *, prefix: str) -> Path:
    prefixed_path = root / cam_name / f"{prefix}_{frame_id:08d}.jpg"
    if prefixed_path.is_file():
        return prefixed_path
    return root / cam_name / f"{frame_id:08d}.jpg"


def _mask_bbox(mask):
    import numpy as np

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _finite_or_nan(shape):
    import numpy as np

    arr = np.empty(shape, dtype=np.float32)
    arr.fill(np.nan)
    return arr


def _run_with_log(log_path: Path, func):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            return func()


def _format_runtime_error(exc: BaseException) -> str:
    text = str(exc)
    if "CUDNN_BACKEND_TENSOR_DESCRIPTOR" in text or "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH" in text:
        return (
            "SAM3D Body inference hit a cuDNN runtime mismatch. "
            "cuDNN is disabled by the MHR wrapper; if this message persists, "
            "run MHR in a clean PyTorch/CUDA environment. "
            f"Original error: {text}"
        )
    return text


def _status(subject: str, message: str) -> None:
    print(f"[mhr:{subject}] {message}", flush=True)


def _save_mhr_params(
    output_path: Path,
    *,
    frame_ids,
    cam_names,
    num_views,
    global_rot,
    body_pose,
    translation,
    shape_params,
    scale_params,
    hand_pose,
    expr_params,
    triangulated,
    loss_total,
    faces,
    checkpointed_frames: int,
    complete: bool,
    reference_cam_names=None,
    fit_diagnostics=None,
) -> None:
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp.npz")
    extra = {} if reference_cam_names is None else {"reference_cam_names": np.asarray(reference_cam_names)}
    extra.update(fit_diagnostics or {})
    extra["estimation_signature"] = np.asarray(os.environ.get("VCI_MHR_ESTIMATION_SIGNATURE", ""))
    np.savez_compressed(
        tmp_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        cam_names=np.asarray(cam_names),
        num_views=num_views,
        global_rot=global_rot,
        body_pose=body_pose,
        translation=translation,
        shape=shape_params,
        scale=scale_params,
        hand_pose=hand_pose,
        expression=expr_params,
        triangulated_keypoints_3d=triangulated,
        loss_total=loss_total,
        faces=faces,
        checkpointed_frames=np.asarray(checkpointed_frames, dtype=np.int32),
        complete=np.asarray(complete, dtype=np.bool_),
        **extra,
    )
    tmp_path.replace(output_path)


def _load_mhr_checkpoint(
    output_path: Path,
    *,
    frame_ids,
    num_views,
    global_rot,
    body_pose,
    translation,
    shape_params,
    scale_params,
    hand_pose,
    expr_params,
    triangulated,
    loss_total,
    reference_cam_names=None,
    fit_diagnostics=None,
) -> int:
    import numpy as np

    if not output_path.is_file():
        return 0
    try:
        with np.load(output_path) as data:
            signature = os.environ.get("VCI_MHR_ESTIMATION_SIGNATURE")
            if signature is not None and str(data.get("estimation_signature", np.asarray("")).item()) != signature:
                return 0
            saved_frame_ids = data["frame_ids"]
            expected_frame_ids = np.asarray(frame_ids, dtype=np.int32)
            if saved_frame_ids.shape != expected_frame_ids.shape or not np.array_equal(saved_frame_ids, expected_frame_ids):
                return 0
            required = {"num_views": num_views, "global_rot": global_rot,
                        "body_pose": body_pose, "translation": translation,
                        "shape": shape_params, "scale": scale_params, "hand_pose": hand_pose,
                        "expression": expr_params, "triangulated_keypoints_3d": triangulated,
                        "loss_total": loss_total}
            if any(key not in data or data[key].shape != target.shape for key, target in required.items()):
                return 0
            checkpointed = int(data.get("checkpointed_frames", np.asarray(0)).item())
            if not 0 <= checkpointed <= len(frame_ids):
                return 0
            if bool(data.get("complete", np.asarray(False)).item()) and checkpointed != len(frame_ids):
                return 0
            for key, target in (
                ("num_views", num_views),
                ("global_rot", global_rot),
                ("body_pose", body_pose),
                ("translation", translation),
                ("shape", shape_params),
                ("scale", scale_params),
                ("hand_pose", hand_pose),
                ("expression", expr_params),
                ("triangulated_keypoints_3d", triangulated),
                ("loss_total", loss_total),
                ("reference_cam_names", reference_cam_names),
            ):
                if target is not None and key in data and data[key].shape == target.shape:
                    target[...] = data[key]
            for key, target in (fit_diagnostics or {}).items():
                if key in data and data[key].shape == target.shape:
                    target[...] = data[key]
            return max(0, min(checkpointed, len(frame_ids)))
    except Exception:
        return 0


def _install_simplecv_triangulate_compat() -> None:
    """Provide the tiny simplecv triangulation API used by sam3d-body-rerun."""
    import numpy as np

    if "simplecv.ops.triangulate" in sys.modules:
        return

    def batch_triangulate(keypoints_2d, projection_matrices, min_views=2):
        keypoints = np.asarray(keypoints_2d, dtype=np.float64)
        projections = np.asarray(projection_matrices, dtype=np.float64)
        if keypoints.ndim != 3 or keypoints.shape[-1] < 3:
            raise ValueError("keypoints_2d must have shape [n_views, n_kpts, 3]")

        n_views, n_kpts, _ = keypoints.shape
        out = np.full((n_kpts, 4), np.nan, dtype=np.float32)
        for kpt_idx in range(n_kpts):
            rows = []
            confidences = []
            for view_idx in range(n_views):
                u, v, conf = keypoints[view_idx, kpt_idx, :3]
                if not np.isfinite(u) or not np.isfinite(v) or conf <= 0:
                    continue
                P = projections[view_idx]
                weight = float(conf)
                rows.append(weight * (u * P[2] - P[0]))
                rows.append(weight * (v * P[2] - P[1]))
                confidences.append(weight)

            if len(confidences) < min_views:
                continue
            _, _, vh = np.linalg.svd(np.stack(rows, axis=0), full_matrices=False)
            point_h = vh[-1]
            if abs(point_h[3]) < 1e-12:
                continue
            point = point_h[:3] / point_h[3]
            out[kpt_idx, :3] = point.astype(np.float32)
            out[kpt_idx, 3] = float(np.mean(confidences))
        return out

    def proj_3d_vectorized(points_3d, projection_matrices):
        points = np.asarray(points_3d, dtype=np.float64)
        projections = np.asarray(projection_matrices, dtype=np.float64)
        points_h = np.concatenate([points[..., :3], np.ones((*points.shape[:-1], 1))], axis=-1)
        projected = np.einsum("vij,kj->vki", projections, points_h)
        return projected[..., :2] / np.maximum(projected[..., 2:3], 1e-12)

    simplecv_mod = sys.modules.setdefault("simplecv", types.ModuleType("simplecv"))
    ops_mod = sys.modules.setdefault("simplecv.ops", types.ModuleType("simplecv.ops"))
    triangulate_mod = types.ModuleType("simplecv.ops.triangulate")
    triangulate_mod.batch_triangulate = batch_triangulate
    triangulate_mod.proj_3d_vectorized = proj_3d_vectorized
    ops_mod.triangulate = triangulate_mod
    simplecv_mod.ops = ops_mod
    sys.modules["simplecv.ops.triangulate"] = triangulate_mod


def _evaluate_mhr_fit(head, rotation, pose, translation, shape, scale, hands,
                      expression, calibration, cameras, observations, triangulated, device):
    """Evaluate saved final parameters, after the optimizer's last update."""
    import numpy as np
    import torch
    from capture_processing.tools.visualize_mhr import camera_mesh_to_world
    from capture_processing.utils.fit_metrics import reprojection_errors

    def tensor(value):
        return torch.from_numpy(np.asarray(value, dtype=np.float32)[None]).to(device)
    with torch.no_grad():
        output = head.mhr_forward(
            global_trans=tensor(np.zeros(3)), global_rot=tensor(rotation),
            body_pose_params=tensor(pose), hand_pose_params=tensor(hands),
            scale_params=tensor(scale), shape_params=tensor(shape),
            expr_params=tensor(expression), return_keypoints=True, return_joint_coords=True)
    keypoints = output[1][0, :70].detach().cpu().numpy()
    world = camera_mesh_to_world(keypoints, translation, calibration[cameras[0]])
    total, count = 0., 0
    for camera_name, detected in zip(cameras, observations):
        camera = calibration[camera_name]
        _, counts, totals = reprojection_errors(world[None], detected[None], camera['K'],
                                                camera['world_to_cam'], detected[None, :, 2])
        total += totals[0]
        count += counts[0]
    valid = np.isfinite(triangulated).all(axis=1) & (triangulated[:, 3] > 0)
    keypoint_error = float(np.linalg.norm(world[valid] - triangulated[valid, :3], axis=1).mean() * 1000) if valid.any() else np.nan
    return {'reprojection_error_px': float(total / count) if count else np.nan,
            'triangulated_keypoint_error_mm': keypoint_error}


def main() -> None:
    args = parse_args()
    repo_src = args.repo_path.expanduser().resolve() / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
    warnings.filterwarnings("ignore", message="Momentum is not enabled")

    import cv2
    import numpy as np
    import torch

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    _install_simplecv_triangulate_compat()

    from sam3d_body.build_models import load_sam_3d_body_hf
    from sam3d_body.ops.optimizer import MultiviewBodyOptimizer, MultiviewOptimizerConfig
    from sam3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

    subject_root = args.subject_root.expanduser().resolve()
    if "VCI_MHR_ESTIMATION_SIGNATURE" not in os.environ:
        from capture_processing.utils.mhr import estimation_signature
        os.environ["VCI_MHR_ESTIMATION_SIGNATURE"] = estimation_signature(
            subject_root, args.repo_path.expanduser().resolve(), args.max_frames, args.iterations
        )
    calibration = _load_calibration(subject_root / "calibration.json")
    cam_names = sorted(calibration)
    frame_ids = _frame_ids(subject_root, cam_names)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise SystemExit("--max-frames must be positive")
        frame_ids = frame_ids[: args.max_frames]
    if args.shard_count <= 0:
        raise SystemExit("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    if args.shard_count > 1:
        frame_ids = frame_ids[args.shard_index :: args.shard_count]
    if not frame_ids:
        raise SystemExit(f"No color frames found under {subject_root / 'images'}")
    mhr_log_path = subject_root / "mhr_runtime.log"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("SAM3D Body estimator currently requires CUDA.")

    print(f"Loading SAM3D Body model from Hugging Face... (details: {mhr_log_path})")
    try:
        sam3d_body_model = _run_with_log(
            mhr_log_path,
            lambda: load_sam_3d_body_hf(repo_id="facebook/sam-3d-body-dinov3")[0],
        )
    except Exception as exc:
        raise SystemExit(
            "Failed to load facebook/sam-3d-body-dinov3. "
            "If this is a Hugging Face access problem, request access at "
            "https://huggingface.co/facebook/sam-3d-body-dinov3, log in at "
            "https://huggingface.co/login, create a token at "
            "https://huggingface.co/settings/tokens, then authenticate this remote "
            "machine with `conda run -n vci-dataset huggingface-cli login` "
            "or by exporting HF_TOKEN before running the pipeline. "
            f"See {mhr_log_path} for details. Original error: {exc}"
        ) from exc
    body_estimator = SAM3DBodyEstimator(sam_3d_body_model=sam3d_body_model)
    print("SAM3D Body model loaded.")

    n_frames = len(frame_ids)
    reference_cam_names = np.full(n_frames, "", dtype=f"U{max(map(len, cam_names), default=1)}")
    global_rot = _finite_or_nan((n_frames, 3))
    body_pose = _finite_or_nan((n_frames, 133))
    translation = _finite_or_nan((n_frames, 3))
    shape_params = _finite_or_nan((n_frames, 45))
    scale_params = _finite_or_nan((n_frames, 28))
    hand_pose = _finite_or_nan((n_frames, 108))
    expr_params = _finite_or_nan((n_frames, 72))
    triangulated = _finite_or_nan((n_frames, 70, 4))
    loss_total = _finite_or_nan((n_frames,))
    num_views = np.zeros((n_frames,), dtype=np.int32)
    fit_diagnostics = {key: _finite_or_nan((n_frames,)) for key in
                       ("reprojection_error_px", "triangulated_keypoint_error_mm")}

    faces = np.asarray(body_estimator.faces, dtype=np.int32)

    start_index = 0 if args.force else _load_mhr_checkpoint(
        args.output,
        frame_ids=frame_ids,
        num_views=num_views,
        global_rot=global_rot,
        body_pose=body_pose,
        translation=translation,
        shape_params=shape_params,
        scale_params=scale_params,
        hand_pose=hand_pose,
        expr_params=expr_params,
        triangulated=triangulated,
        loss_total=loss_total,
        fit_diagnostics=fit_diagnostics,
        reference_cam_names=reference_cam_names,
    )
    if start_index >= n_frames:
        _status(subject_root.name, f"finalizing fully checkpointed MHR parameters -> {args.output}")
    if start_index > 0:
        _status(subject_root.name, f"resuming MHR from checkpointed frame {start_index}/{n_frames} -> {args.output}")

    for out_idx in range(start_index, n_frames):
        frame_id = frame_ids[out_idx]
        predictions = []
        keypoints_2d = []
        p_mats = []
        first_world_to_cam = None
        used_cams = []

        for cam_name in cam_names:
            image_path = _frame_path(subject_root / "images", cam_name, frame_id, prefix="color")
            mask_path = _frame_path(subject_root / "masks", cam_name, frame_id, prefix="mask")
            if not image_path.is_file() or not mask_path.is_file():
                continue

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mask is None:
                continue
            binary_mask = (mask > 0).astype(np.float32)
            bbox = _mask_bbox(binary_mask)
            if bbox is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                pred_list = body_estimator.process_one_image(
                    rgb_hw3=rgb,
                    xyxy=bbox.reshape(1, 4),
                    masks=binary_mask.reshape(1, *binary_mask.shape),
                    masks_score=np.ones((1,), dtype=np.float32),
                    K_33=calibration[cam_name]["K"],
                )
            except RuntimeError as exc:
                raise SystemExit(
                    f"SAM3D Body inference failed on frame {frame_id}, camera {cam_name}. "
                    f"{_format_runtime_error(exc)}"
                ) from exc
            if not pred_list:
                continue

            pred = pred_list[0]
            predictions.append(pred)
            conf = np.ones((pred.pred_keypoints_2d.shape[0], 1), dtype=np.float32)
            keypoints_2d.append(np.concatenate([pred.pred_keypoints_2d.astype(np.float32), conf], axis=-1))
            p_mats.append(calibration[cam_name]["P"])
            if first_world_to_cam is None:
                first_world_to_cam = calibration[cam_name]["world_to_cam"]
            used_cams.append(cam_name)

        num_views[out_idx] = len(predictions)
        if not predictions:
            print(f"[warn] frame {frame_id}: no valid camera predictions", flush=True)
            _save_mhr_params(
                args.output,
                frame_ids=frame_ids,
                cam_names=cam_names,
                num_views=num_views,
                global_rot=global_rot,
                body_pose=body_pose,
                translation=translation,
                shape_params=shape_params,
                scale_params=scale_params,
                hand_pose=hand_pose,
                expr_params=expr_params,
                triangulated=triangulated,
                loss_total=loss_total,
                fit_diagnostics=fit_diagnostics,
                faces=faces,
                reference_cam_names=reference_cam_names,
                checkpointed_frames=out_idx + 1,
                complete=False,
            )
            _status(subject_root.name, f"checkpoint saved through frame {frame_id} ({out_idx + 1}/{n_frames}) -> {args.output}")
            continue

        reference_cam_names[out_idx] = used_cams[0]
        base = predictions[0]
        shape_params[out_idx] = base.shape_params.astype(np.float32)
        scale_params[out_idx] = base.scale_params.astype(np.float32)
        hand_pose[out_idx] = base.hand_pose_params.astype(np.float32)
        expr_params[out_idx] = base.expr_params.astype(np.float32)

        if len(predictions) >= 2:
            assert first_world_to_cam is not None
            optimizer = MultiviewBodyOptimizer(
                projection_matrices=np.stack(p_mats, axis=0).astype(np.float32),
                first_world_T_cam=np.linalg.inv(first_world_to_cam).astype(np.float32),
                mhr_head=body_estimator.model.head_pose,
                config=MultiviewOptimizerConfig(num_iterations=args.iterations, learning_rate=0.01),
                device=device,
            )
            with torch.enable_grad():
                result = optimizer.optimize_frame(
                    initial_predictions=predictions,
                    keypoints_2d_per_view=np.stack(keypoints_2d, axis=0).astype(np.float32),
                )
            global_rot[out_idx] = result.global_rot.astype(np.float32)
            body_pose[out_idx] = result.body_pose.astype(np.float32)
            translation[out_idx] = result.translation.astype(np.float32)
            triangulated[out_idx] = result.triangulated_kpts_3d.astype(np.float32)
            loss_total[out_idx] = float(result.final_loss.total.detach().cpu())
            print(f"frame {frame_id}: optimized MHR from {len(predictions)} view(s): {used_cams}", flush=True)
        else:
            global_rot[out_idx] = base.global_rot.astype(np.float32)
            body_pose[out_idx] = base.body_pose_params.astype(np.float32)
            translation[out_idx] = base.pred_cam_t.astype(np.float32)
            print(f"frame {frame_id}: single-view MHR fallback from {used_cams[0]}", flush=True)

        residuals = _evaluate_mhr_fit(
            body_estimator.model.head_pose, global_rot[out_idx], body_pose[out_idx],
            translation[out_idx], shape_params[out_idx], scale_params[out_idx],
            hand_pose[out_idx], expr_params[out_idx], calibration, used_cams,
            np.stack(keypoints_2d), triangulated[out_idx], device)
        for name, value in residuals.items():
            fit_diagnostics[name][out_idx] = value

        _save_mhr_params(
            args.output,
            frame_ids=frame_ids,
            cam_names=cam_names,
            num_views=num_views,
            global_rot=global_rot,
            body_pose=body_pose,
            translation=translation,
            shape_params=shape_params,
            scale_params=scale_params,
            hand_pose=hand_pose,
            expr_params=expr_params,
            triangulated=triangulated,
            loss_total=loss_total,
            fit_diagnostics=fit_diagnostics,
            faces=faces,
            reference_cam_names=reference_cam_names,
            checkpointed_frames=out_idx + 1,
            complete=False,
        )
        _status(subject_root.name, f"checkpoint saved through frame {frame_id} ({out_idx + 1}/{n_frames}) -> {args.output}")

    _save_mhr_params(
        args.output,
        frame_ids=frame_ids,
        cam_names=cam_names,
        num_views=num_views,
        global_rot=global_rot,
        body_pose=body_pose,
        translation=translation,
        shape_params=shape_params,
        scale_params=scale_params,
        hand_pose=hand_pose,
        expr_params=expr_params,
        triangulated=triangulated,
        loss_total=loss_total,
        fit_diagnostics=fit_diagnostics,
        faces=faces,
        reference_cam_names=reference_cam_names,
        checkpointed_frames=n_frames,
        complete=True,
    )
    _status(subject_root.name, f"MHR parameters complete -> {args.output}")


if __name__ == "__main__":
    main()
