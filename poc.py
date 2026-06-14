"""Proof-of-concept reproduction of World Tracing's core claim.

The paper's central representation claim: a single forward pass of the
flow-matching diffusion transformer predicts, for every input pixel, an
ordered front-to-back stack of L camera-space XYZ points that covers the
visible surface (layer 0) AND the occluded surfaces behind it (deeper
layers). See README "What you get back" and Sec 4.1 of the paper.

This script loads the released r75b object model (1.7B params, 504x504,
6 layers), runs ``inference_diffusion`` on the shipped object test images
with a single deterministic seed, and measures whether the prediction
actually exhibits that structure. It does NOT use rerun; instead it dumps
machine-readable evidence so the claim can be judged numerically:

  * num_layers and output tensor shape (the representation is multilayer)
  * layer-0 metric depth + recovered horizontal FoV (visible surface is
    a faithful, camera-consistent pointmap; no external pose estimator)
  * front-to-back monotonicity fraction (deeper layers lie behind nearer
    ones, the front-to-back ordering the paper enforces)
  * generated occluded-surface "thickness": per-ray (z_last - z_0) > eps,
    i.e. deeper layers are genuinely behind the visible surface, not copies

The previous run reported `strict_monotonic_frac = 0.115` while
`occluded_ray_frac = 0.91` on the same images, which is suspicious: only
~11% of rays are strictly monotonic front-to-back, yet 91% of rays carry
real occluded geometry. ``wt/data.py``'s docstring explicitly notes that
``alpha_erode_px`` "helps suppress deep-layer plume artifacts caused by
over-segmented matting / SAM masks". So this PoC sweeps
``alpha_erode_px ∈ {0, 2, 4}`` per image (re-running ``inference_diffusion``
with the same SEED each time) to test whether the low monotonicity lives
at the silhouette boundary (a few stray plume pixels dragging the strict
fraction down) or is intrinsic to the 6-layer prediction.

Artifacts written to .openresearch/artifacts/:
  * EVAL.md      -- human-readable summary table (per-erode columns)
  * metrics.json -- per-image (per-erode) + aggregate metrics (CLI-readable)
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

CONFIG = "r75b"
CKPT = "r75b"  # bare config name -> public HF repo haoz19/object-model-6layer
SEED = 42
# Thickness threshold (meters): a ray's back-most predicted layer must be at
# least this far behind its visible surface to count as "real occluded
# geometry" rather than the depth-filling forward-copy of layer 0.
THICKNESS_EPS = 0.02
# Tolerance (meters) for calling an adjacent-layer depth step non-decreasing.
MONO_TOL = 0.005
# Plume-artifact hypothesis sweep: erode the foreground silhouette by this
# many pixels in each preprocessing pass. ``0`` reproduces the original
# behaviour; ``2``/``4`` peel off boundary pixels where over-segmented mattes
# tend to produce the "deep-layer plume" the wt/data.py docstring warns about.
ALPHA_ERODE_PX = [0, 2, 4]

IMAGES = [
    "examples/test_images/object/obj014_leather_briefcase.png",
    "examples/test_images/object/obj063_trex_dinosaur.png",
    "examples/test_images/object/obj070_red_fox.png",
    "examples/test_images/object/obj040_leather_armchair.png",
]

ARTIFACT_DIR = Path(".openresearch/artifacts")


def analyze(xyz: np.ndarray, mask: np.ndarray) -> dict:
    """Compute multilayer-geometry evidence metrics for one prediction.

    Args:
        xyz:  [L, H, W, 3] camera-space XYZ (metric meters for r75b).
        mask: [L, H, W] bool valid mask (layer-0 silhouette, AND-accumulated).
    """
    L = xyz.shape[0]
    z = xyz[..., 2]  # [L, H, W]
    valid0 = mask[0]  # visible-surface silhouette
    n0 = int(valid0.sum())

    # Per-layer mean depth over the layer-0 silhouette (front-to-back profile).
    per_layer_mean_z = [float(z[l][valid0].mean()) for l in range(L)]
    # Aggregate front-to-back ordering: is the mean-depth profile itself
    # non-decreasing (each layer's mean depth >= the previous layer's)?
    profile_monotonic = all(
        per_layer_mean_z[l + 1] >= per_layer_mean_z[l] - 1e-4 for l in range(L - 1)
    )

    zr = z.reshape(L, -1)[:, valid0.reshape(-1)]  # [L, N]
    diffs = zr[1:] - zr[:-1]  # [L-1, N]
    # Soft front-to-back ordering (matches the paper's *soft* monotonicity
    # penalty, not a hard constraint): fraction of all adjacent layer
    # transitions, over all rays, that are non-decreasing within tolerance.
    pairwise_nondecreasing = float((diffs >= -MONO_TOL).mean())
    # Stricter view: fraction of rays non-decreasing across *every* pair.
    strict_monotonic = float((diffs >= -MONO_TOL).all(axis=0).mean())

    # Generated occluded geometry: thickness = z_last - z_0 per ray.
    thickness = zr[-1] - zr[0]  # [N]
    thick_frac = float((thickness > THICKNESS_EPS).mean())
    mean_thickness = float(thickness.mean())
    median_thickness = float(np.median(thickness))

    # Layer-0 visible surface depth (metric).
    z0 = zr[0]
    return {
        "num_layers": L,
        "valid_pixels_layer0": n0,
        "layer0_depth_mean_m": float(z0.mean()),
        "layer0_depth_median_m": float(np.median(z0)),
        "per_layer_mean_z_m": per_layer_mean_z,
        "profile_monotonic": profile_monotonic,
        "pairwise_nondecreasing_frac": pairwise_nondecreasing,
        "strict_monotonic_frac": strict_monotonic,
        "occluded_thickness_mean_m": mean_thickness,
        "occluded_thickness_median_m": median_thickness,
        "occluded_ray_frac": thick_frac,
    }


def run_one(model, cfg, device, autocast_ctx, rgba, erode_px: int) -> dict:
    """Preprocess with ``alpha_erode_px=erode_px`` and run one diffusion pass.

    The SEED is re-applied before every call so the only thing that varies
    across the sweep is the silhouette boundary fed to the model.
    """
    rgb_t, mask_t, intr_t = preprocess_rgba_for_model(
        rgba,
        image_size=cfg["image_size"],
        num_layers=cfg["model_kwargs"]["num_layers"],
        center_crop=True,
        alpha_erode_px=erode_px,
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
    m["alpha_erode_px"] = erode_px
    m["recovered_fov_x_deg"] = float(fov_x)
    m["inference_s"] = round(dt, 2)
    m["xyz_shape"] = list(xyz_pred.shape)
    return m


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
        rgba = load_rgba_image(img_path, auto_alpha=True)

        by_erode: dict[int, dict] = {}
        for erode_px in ALPHA_ERODE_PX:
            print(f"[poc]  -- alpha_erode_px={erode_px}")
            m = run_one(model, cfg, device, autocast_ctx, rgba, erode_px)
            by_erode[erode_px] = m
            print(
                f"[poc]     mono={m['strict_monotonic_frac']:.3f} "
                f"pairwise_nondecr={m['pairwise_nondecreasing_frac']:.3f} "
                f"occluded_rays={m['occluded_ray_frac']:.3f} "
                f"thickness={m['occluded_thickness_mean_m']:.3f}m "
                f"L0_depth={m['layer0_depth_mean_m']:.3f}m "
                f"fov_x={m['recovered_fov_x_deg']:.1f}deg "
                f"({m['inference_s']:.1f}s)"
            )
        results.append({"image": name, "by_erode": by_erode})

    # ---- Aggregate (per erode value) ----
    by_erode_agg: dict[int, dict] = {}
    for e in ALPHA_ERODE_PX:
        rs = [r["by_erode"][e] for r in results]
        by_erode_agg[e] = {
            "alpha_erode_px": e,
            "all_profiles_monotonic": bool(all(r["profile_monotonic"] for r in rs)),
            "mean_strict_monotonic_frac": round(
                float(np.mean([r["strict_monotonic_frac"] for r in rs])), 4
            ),
            "mean_pairwise_nondecreasing_frac": round(
                float(np.mean([r["pairwise_nondecreasing_frac"] for r in rs])), 4
            ),
            "mean_occluded_ray_frac": round(
                float(np.mean([r["occluded_ray_frac"] for r in rs])), 4
            ),
            "mean_occluded_thickness_m": round(
                float(np.mean([r["occluded_thickness_mean_m"] for r in rs])), 4
            ),
            "mean_recovered_fov_x_deg": round(
                float(np.mean([r["recovered_fov_x_deg"] for r in rs])), 2
            ),
        }

    agg = {
        "config": CONFIG,
        "params_billion": round(n_params, 3),
        "image_size": cfg["image_size"],
        "num_layers": cfg["model_kwargs"]["num_layers"],
        "num_steps": cfg["inference_kwargs"]["num_steps"],
        "seed": SEED,
        "thickness_eps_m": THICKNESS_EPS,
        "mono_tol_m": MONO_TOL,
        "alpha_erode_px_sweep": ALPHA_ERODE_PX,
        "n_images": len(results),
        "by_erode": by_erode_agg,
        "per_image": results,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(agg, indent=2))

    # ---- EVAL.md ----
    lines = [
        "# World Tracing r75b -- multilayer-geometry PoC (alpha_erode_px sweep)",
        "",
        f"- model: {CONFIG} ({agg['params_billion']}B params), "
        f"{agg['image_size']}x{agg['image_size']}, L={agg['num_layers']} layers, "
        f"{agg['num_steps']} ODE steps, seed {SEED}",
        f"- images: {agg['n_images']} shipped object test images",
        f"- alpha_erode_px sweep: {ALPHA_ERODE_PX} (re-runs `inference_diffusion` "
        f"with the same SEED for each value)",
        "",
        "Plume-artifact hypothesis: ``wt/data.py`` says `alpha_erode_px` helps "
        "suppress deep-layer plume artifacts from over-segmented mattes. If the "
        "previous run's low strict-monotonicity (0.115) was caused by a thin "
        "layer of plume pixels at the silhouette boundary, eroding the mask "
        "should make `mono` jump while `occluded_ray_frac` and `thickness` stay "
        "broadly the same. If `mono` stays low after erosion, the non-monotonicity "
        "is intrinsic to the 6-layer prediction.",
        "",
        "## Core-claim metrics (per image, per alpha_erode_px)",
        "",
    ]
    # Header: one image column + (mono, occluded_ray_frac, thickness) per erode.
    header_cells = ["image"]
    for e in ALPHA_ERODE_PX:
        header_cells += [f"mono@e={e}", f"occl@e={e}", f"thick@e={e} (m)"]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for r in results:
        row = [r["image"]]
        for e in ALPHA_ERODE_PX:
            m = r["by_erode"][e]
            row += [
                f"{m['strict_monotonic_frac']:.3f}",
                f"{m['occluded_ray_frac']:.3f}",
                f"{m['occluded_thickness_mean_m']:.3f}",
            ]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Aggregate (mean over images, per alpha_erode_px)",
        "",
        "| alpha_erode_px | mean mono (strict) | mean pairwise nondecr | "
        "mean occluded rays | mean thickness (m) | profiles mono on all | "
        "mean fov_x (deg) |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in ALPHA_ERODE_PX:
        a = by_erode_agg[e]
        lines.append(
            f"| {e} | {a['mean_strict_monotonic_frac']:.3f} | "
            f"{a['mean_pairwise_nondecreasing_frac']:.3f} | "
            f"{a['mean_occluded_ray_frac']:.3f} | "
            f"{a['mean_occluded_thickness_m']:.3f} | "
            f"{a['all_profiles_monotonic']} | "
            f"{a['mean_recovered_fov_x_deg']:.1f} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        f"- `mono` is the strict per-ray monotonicity fraction "
        f"(tol {MONO_TOL} m): fraction of rays whose z is non-decreasing "
        "across *every* adjacent layer pair.",
        f"- `occl` is the occluded-ray fraction (thickness > {THICKNESS_EPS} m): "
        "rays where the model placed real geometry behind the visible surface.",
        "- `thick` is the mean per-ray (z_last - z_0) in meters.",
        "- A monotonicity jump as `alpha_erode_px` grows from 0 -> 2 -> 4, with "
        "`occl`/`thick` roughly preserved, supports the silhouette-boundary "
        "plume-artifact hypothesis. Monotonicity staying low (~0.1) across the "
        "sweep means the non-monotonicity is intrinsic to the 6-layer "
        "prediction, not a preprocessing artifact.",
    ]
    (ARTIFACT_DIR / "EVAL.md").write_text("\n".join(lines))
    print("\n[poc] wrote .openresearch/artifacts/EVAL.md and metrics.json")
    for e in ALPHA_ERODE_PX:
        a = by_erode_agg[e]
        print(
            f"[poc] erode={e}: mono={a['mean_strict_monotonic_frac']:.3f} "
            f"pairwise_nondecr={a['mean_pairwise_nondecreasing_frac']:.3f} "
            f"occluded_rays={a['mean_occluded_ray_frac']:.3f} "
            f"thickness={a['mean_occluded_thickness_m']:.3f}m "
            f"fov_x={a['mean_recovered_fov_x_deg']:.1f}deg"
        )


if __name__ == "__main__":
    main()
