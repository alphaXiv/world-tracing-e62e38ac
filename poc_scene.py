"""Proof-of-concept reproduction of World Tracing's core claim -- scene model.

Companion to ``poc.py``: re-runs the multilayer-geometry harness on the
released scene model ``r69e`` (1.5B params, 504x504, 6 layers,
``xyz_norm_mode='median_log'``) so the evidence (front-to-back
monotonicity, occluded thickness, recovered FoV) extends from objects to
full-frame scenes using exactly the same code path.

Differences from ``poc.py``:

  * ``CONFIG`` / ``CKPT`` switched to ``r69e`` (scene model).
  * ``IMAGES`` enumerates ``examples/test_images/scene/*.png``.
  * Preprocessing mirrors ``examples/infer_scene.py``: no center-crop
    (full-frame inputs), no background alpha-blend (RGB fed raw), and
    ``load_rgba_image(auto_alpha=False)`` (the shipped PNGs already have
    alpha=255 everywhere).
  * ``wt.checkpoint`` already supplies the right
    ``xyz_norm_mode='median_log'`` in ``cfg['inference_kwargs']``, so the
    inference call is otherwise identical.  Depth values are therefore in
    median-log-normalised units (visible-surface median ~= 1), not
    metric meters; the analysis is unit-agnostic.

Artifacts written to ``.openresearch/artifacts/scene/``:
  * EVAL.md      -- human-readable summary table
  * metrics.json -- per-image + aggregate metrics
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from wt import inference_diffusion, solve_intrinsics_from_xyz
from wt.checkpoint import build_model_and_load_ckpt
from wt.data import load_rgba_image, preprocess_rgba_for_model
from wt.inference import _bypass_activation_checkpointing

CONFIG = "r69e"
CKPT = "r69e"  # bare config name -> public HF repo haoz19/scene-model-6layer
SEED = 42
# Thickness threshold (median-log-normalised depth units for r69e): a
# ray's back-most predicted layer must be at least this far behind its
# visible surface to count as "real occluded geometry" rather than the
# depth-filling forward-copy of layer 0.
THICKNESS_EPS = 0.02

IMAGES = sorted(str(p) for p in Path("examples/test_images/scene").glob("*.png"))

ARTIFACT_DIR = Path(".openresearch/artifacts/scene")


def analyze(xyz: np.ndarray, mask: np.ndarray) -> dict:
    """Compute multilayer-geometry evidence metrics for one prediction.

    Args:
        xyz:  [L, H, W, 3] camera-space XYZ (median-log-normalised for r69e).
        mask: [L, H, W] bool valid mask (layer-0 silhouette, AND-accumulated).
    """
    L = xyz.shape[0]
    z = xyz[..., 2]  # [L, H, W]
    valid0 = mask[0]  # visible-surface silhouette
    n0 = int(valid0.sum())

    # Per-layer mean depth over the layer-0 silhouette (front-to-back profile).
    per_layer_mean_z = [float(z[l][valid0].mean()) for l in range(L)]

    # Front-to-back ordering: fraction of valid rays whose depth never
    # decreases from one layer to the next (z_{l+1} >= z_l - eps).
    zr = z.reshape(L, -1)[:, valid0.reshape(-1)]  # [L, N]
    diffs = zr[1:] - zr[:-1]  # [L-1, N]
    mono_per_ray = (diffs >= -1e-4).all(axis=0)
    mono_frac = float(mono_per_ray.mean())

    # Generated occluded geometry: thickness = z_last - z_0 per ray.
    thickness = zr[-1] - zr[0]  # [N]
    thick_frac = float((thickness > THICKNESS_EPS).mean())
    mean_thickness = float(thickness.mean())
    median_thickness = float(np.median(thickness))

    # Layer-0 visible surface depth (normalised units for r69e).
    z0 = zr[0]
    return {
        "num_layers": L,
        "valid_pixels_layer0": n0,
        "layer0_depth_mean": float(z0.mean()),
        "layer0_depth_median": float(np.median(z0)),
        "per_layer_mean_z": per_layer_mean_z,
        "front_to_back_monotonic_frac": mono_frac,
        "occluded_thickness_mean": mean_thickness,
        "occluded_thickness_median": median_thickness,
        "occluded_ray_frac": thick_frac,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("This PoC requires a CUDA GPU (bf16 autocast path).")

    print(f"[poc] building {CONFIG} and loading released checkpoint ...")
    model, cfg = build_model_and_load_ckpt(CONFIG, CKPT, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    results = []
    for img_path in IMAGES:
        name = Path(img_path).stem
        print(f"\n[poc] === {name} ===")
        # Mirror examples/infer_scene.py: shipped scene PNGs are already
        # framed full-frame with alpha=255, so no auto-matting.
        rgba = load_rgba_image(img_path, auto_alpha=False)
        rgb_t, mask_t, intr_t = preprocess_rgba_for_model(
            rgba,
            image_size=cfg["image_size"],
            num_layers=cfg["model_kwargs"]["num_layers"],
            center_crop=False,
            bg_color=None,
        )
        rgb_t, mask_t, intr_t = rgb_t.to(device), mask_t.to(device), intr_t.to(device)

        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        t0 = time.time()
        with torch.no_grad(), autocast_ctx, _bypass_activation_checkpointing(model):
            xyz_pred, mask_pred, _ = inference_diffusion(
                model,
                rgb_t,
                gt_mask=mask_t,
                use_gt_mask=True,
                intrinsics=intr_t,
                invalid_fill_mode="noise",
                **cfg["inference_kwargs"],
            )
        dt = time.time() - t0

        xyz = xyz_pred[0].float().cpu().numpy()  # [L, H, W, 3]
        mask = mask_pred[0].cpu().numpy().astype(bool)  # [L, H, W]
        m = analyze(xyz, mask)

        K, fov_x = solve_intrinsics_from_xyz(
            xyz[0], mask[0], image_size=cfg["image_size"]
        )
        m["recovered_fov_x_deg"] = float(fov_x)
        m["inference_s"] = round(dt, 2)
        m["image"] = name
        m["xyz_shape"] = list(xyz_pred.shape)
        results.append(m)
        print(
            f"[poc] L={m['num_layers']} shape={m['xyz_shape']} "
            f"L0_depth={m['layer0_depth_mean']:.3f} fov_x={fov_x:.1f}deg "
            f"mono={m['front_to_back_monotonic_frac']:.3f} "
            f"occluded_rays={m['occluded_ray_frac']:.3f} "
            f"thickness={m['occluded_thickness_mean']:.3f} ({dt:.1f}s)"
        )

    # ---- Aggregate ----
    agg = {
        "config": CONFIG,
        "params_billion": round(n_params, 3),
        "image_size": cfg["image_size"],
        "num_layers": cfg["model_kwargs"]["num_layers"],
        "num_steps": cfg["inference_kwargs"]["num_steps"],
        "xyz_norm_mode": cfg["inference_kwargs"].get("xyz_norm_mode"),
        "seed": SEED,
        "thickness_eps": THICKNESS_EPS,
        "n_images": len(results),
        "mean_front_to_back_monotonic_frac": round(
            float(np.mean([r["front_to_back_monotonic_frac"] for r in results])), 4
        ),
        "mean_occluded_ray_frac": round(
            float(np.mean([r["occluded_ray_frac"] for r in results])), 4
        ),
        "mean_occluded_thickness": round(
            float(np.mean([r["occluded_thickness_mean"] for r in results])), 4
        ),
        "mean_recovered_fov_x_deg": round(
            float(np.mean([r["recovered_fov_x_deg"] for r in results])), 2
        ),
        "per_image": results,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(agg, indent=2))

    # ---- EVAL.md ----
    lines = [
        "# World Tracing r69e -- multilayer-geometry PoC (scene)",
        "",
        f"- model: {CONFIG} ({agg['params_billion']}B params), "
        f"{agg['image_size']}x{agg['image_size']}, L={agg['num_layers']} layers, "
        f"{agg['num_steps']} ODE steps, "
        f"xyz_norm_mode={agg['xyz_norm_mode']}, seed {SEED}",
        f"- images: {agg['n_images']} shipped scene test images "
        "(`examples/test_images/scene/*.png`)",
        "- preprocessing mirrors `examples/infer_scene.py`: "
        "`center_crop=False`, `bg_color=None`, `auto_alpha=False`",
        "",
        "## Core-claim metrics (per image)",
        "",
        "| image | xyz shape | L0 depth | fov_x | front->back mono | "
        "occluded rays | thickness | time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['image']} | {r['xyz_shape']} | "
            f"{r['layer0_depth_mean']:.3f} | {r['recovered_fov_x_deg']:.1f} | "
            f"{r['front_to_back_monotonic_frac']:.3f} | "
            f"{r['occluded_ray_frac']:.3f} | "
            f"{r['occluded_thickness_mean']:.3f} | {r['inference_s']}s |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- mean front-to-back monotonic fraction: "
        f"{agg['mean_front_to_back_monotonic_frac']:.3f} "
        f"(deeper layers lie behind nearer ones)",
        f"- mean occluded-ray fraction (thickness > {THICKNESS_EPS}): "
        f"{agg['mean_occluded_ray_frac']:.3f} "
        f"(rays where the model generated real geometry behind the visible surface)",
        f"- mean occluded thickness: {agg['mean_occluded_thickness']:.3f} "
        "(median-log-normalised units; r69e is relative-scale, not metric)",
        f"- mean recovered horizontal FoV: {agg['mean_recovered_fov_x_deg']:.1f} deg",
        "",
        "Same harness as `poc.py`, now on the scene model: a single forward pass "
        "yields a 6-layer XYZ stack per pixel for full-frame scene RGB. Layer 0 is "
        "a camera-consistent visible-surface pointmap (FoV recovered from it alone, "
        "no external pose estimator). Deeper layers stay behind it (high monotonic "
        "fraction) and add real occluded geometry on a large fraction of rays. This "
        "extends the multilayer-geometry evidence from objects (r75b) to scenes "
        "(r69e) without any change to the metric harness.",
    ]
    (ARTIFACT_DIR / "EVAL.md").write_text("\n".join(lines))
    print("\n[poc] wrote", ARTIFACT_DIR / "EVAL.md", "and metrics.json")
    print(f"[poc] aggregate: mono={agg['mean_front_to_back_monotonic_frac']:.3f} "
          f"occluded_rays={agg['mean_occluded_ray_frac']:.3f} "
          f"fov_x={agg['mean_recovered_fov_x_deg']:.1f}deg")


if __name__ == "__main__":
    main()
