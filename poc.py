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

In addition, this version sweeps the only inference-time compute knob --
``cfg['inference_kwargs']['num_steps']`` (the ODE step count of the
flow-matching sampler) -- over {5, 10, 20, 40} on the same four object
images with the same seed, so we can see how the multilayer-geometry
metrics scale with sampler compute and judge whether the released 20-step
default is over- or under-spending compute for the multilayer claim.

Artifacts written to .openresearch/artifacts/:
  * EVAL.md      -- human-readable summary table
  * metrics.json -- per-image + aggregate metrics (text, CLI-readable)
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

# Sweep the only inference-time compute knob (flow-matching ODE step count).
# The released config default is 20.
NUM_STEPS_SWEEP = [5, 10, 20, 40]

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

    # Layer-0 visible surface depth (metric).
    z0 = zr[0]
    return {
        "num_layers": L,
        "valid_pixels_layer0": n0,
        "layer0_depth_mean_m": float(z0.mean()),
        "layer0_depth_median_m": float(np.median(z0)),
        "per_layer_mean_z_m": per_layer_mean_z,
        "front_to_back_monotonic_frac": mono_frac,
        "occluded_thickness_mean_m": mean_thickness,
        "occluded_thickness_median_m": median_thickness,
        "occluded_ray_frac": thick_frac,
    }


def run_one(model, cfg, device, autocast_ctx, img_path: str) -> dict:
    """Run inference_diffusion on one image with the *current* cfg."""
    name = Path(img_path).stem
    rgba = load_rgba_image(img_path, auto_alpha=True)
    rgb_t, mask_t, intr_t = preprocess_rgba_for_model(
        rgba,
        image_size=cfg["image_size"],
        num_layers=cfg["model_kwargs"]["num_layers"],
        center_crop=True,
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

    sweeps = []
    for num_steps in NUM_STEPS_SWEEP:
        # Override the only inference-time compute knob before each call.
        cfg["inference_kwargs"]["num_steps"] = num_steps
        print(f"\n[poc] === sweep: num_steps={num_steps} ===")

        results = []
        for img_path in IMAGES:
            name = Path(img_path).stem
            print(f"[poc] [num_steps={num_steps}] {name}")
            m = run_one(model, cfg, device, autocast_ctx, img_path)
            results.append(m)
            print(
                f"[poc]   L={m['num_layers']} shape={m['xyz_shape']} "
                f"L0_depth={m['layer0_depth_mean_m']:.3f}m "
                f"fov_x={m['recovered_fov_x_deg']:.1f}deg "
                f"mono={m['front_to_back_monotonic_frac']:.3f} "
                f"occluded_rays={m['occluded_ray_frac']:.3f} "
                f"thickness={m['occluded_thickness_mean_m']:.3f}m "
                f"({m['inference_s']}s)"
            )

        agg = {
            "num_steps": num_steps,
            "n_images": len(results),
            "mean_front_to_back_monotonic_frac": round(
                float(np.mean([r["front_to_back_monotonic_frac"] for r in results])), 4
            ),
            "mean_occluded_ray_frac": round(
                float(np.mean([r["occluded_ray_frac"] for r in results])), 4
            ),
            "mean_occluded_thickness_m": round(
                float(np.mean([r["occluded_thickness_mean_m"] for r in results])), 4
            ),
            "mean_layer0_depth_m": round(
                float(np.mean([r["layer0_depth_mean_m"] for r in results])), 4
            ),
            "mean_recovered_fov_x_deg": round(
                float(np.mean([r["recovered_fov_x_deg"] for r in results])), 2
            ),
            "mean_inference_s": round(
                float(np.mean([r["inference_s"] for r in results])), 2
            ),
            "per_image": results,
        }
        sweeps.append(agg)

    out = {
        "config": CONFIG,
        "params_billion": round(n_params, 3),
        "image_size": cfg["image_size"],
        "num_layers": cfg["model_kwargs"]["num_layers"],
        "num_steps_sweep": NUM_STEPS_SWEEP,
        "seed": SEED,
        "thickness_eps_m": THICKNESS_EPS,
        "sweeps": sweeps,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(out, indent=2))

    # ---- EVAL.md ----
    lines = [
        "# World Tracing r75b -- multilayer-geometry PoC (num_steps sweep)",
        "",
        f"- model: {CONFIG} ({out['params_billion']}B params), "
        f"{out['image_size']}x{out['image_size']}, L={out['num_layers']} layers, "
        f"seed {SEED}",
        f"- images: {len(IMAGES)} shipped object test images",
        f"- sweep: num_steps in {NUM_STEPS_SWEEP} "
        f"(overrides cfg['inference_kwargs']['num_steps']; released default = 20)",
        "",
        "## Aggregate vs. num_steps",
        "",
        "| num_steps | mean front->back mono | mean occluded rays | "
        "mean thickness (m) | mean L0 depth (m) | mean fov_x (deg) | mean time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sweeps:
        lines.append(
            f"| {s['num_steps']} | "
            f"{s['mean_front_to_back_monotonic_frac']:.3f} | "
            f"{s['mean_occluded_ray_frac']:.3f} | "
            f"{s['mean_occluded_thickness_m']:.3f} | "
            f"{s['mean_layer0_depth_m']:.3f} | "
            f"{s['mean_recovered_fov_x_deg']:.1f} | "
            f"{s['mean_inference_s']:.2f} |"
        )

    lines += ["", "## Per-image metrics by num_steps", ""]
    for s in sweeps:
        lines += [
            f"### num_steps = {s['num_steps']}",
            "",
            "| image | xyz shape | L0 depth (m) | fov_x | front->back mono | "
            "occluded rays | thickness (m) | time |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in s["per_image"]:
            lines.append(
                f"| {r['image']} | {r['xyz_shape']} | "
                f"{r['layer0_depth_mean_m']:.3f} | "
                f"{r['recovered_fov_x_deg']:.1f} | "
                f"{r['front_to_back_monotonic_frac']:.3f} | "
                f"{r['occluded_ray_frac']:.3f} | "
                f"{r['occluded_thickness_mean_m']:.3f} | {r['inference_s']}s |"
            )
        lines.append("")

    lines += [
        "## What this sweep isolates",
        "",
        f"All other knobs (seed={SEED}, images, model, layers, image size, "
        f"thickness eps={THICKNESS_EPS} m) are held fixed; only the "
        "flow-matching ODE step count varies. The front-to-back monotonic "
        "fraction, occluded-ray fraction, and mean occluded thickness across "
        "num_steps therefore directly show how the multilayer-geometry claim "
        "scales with inference-time sampler compute, and tells us whether the "
        "released 20-step default is over- or under-spending compute relative "
        "to the cheaper (5, 10) and more expensive (40) settings.",
    ]
    (ARTIFACT_DIR / "EVAL.md").write_text("\n".join(lines))
    print("\n[poc] wrote .openresearch/artifacts/EVAL.md and metrics.json")
    for s in sweeps:
        print(
            f"[poc] num_steps={s['num_steps']}: "
            f"mono={s['mean_front_to_back_monotonic_frac']:.3f} "
            f"occluded_rays={s['mean_occluded_ray_frac']:.3f} "
            f"thickness={s['mean_occluded_thickness_m']:.3f}m "
            f"mean_time={s['mean_inference_s']:.2f}s"
        )


if __name__ == "__main__":
    main()
