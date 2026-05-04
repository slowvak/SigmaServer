"""SigmaServer — AI inference server for the SIGMA medical image viewer.

Runs on a GPU machine (DGX Spark, cloud GPU instance, or local GPU).
Loads models once at startup and exposes them via HTTP API.

Endpoints:
    GET  /models         — list available AI tools with metadata
    POST /predict        — run model inference on a NIfTI volume
    GET  /health         — health check + GPU info

Usage:
    python server.py [--host 0.0.0.0] [--port 8080]

SIGMA image server discovers this server via the "ai.server" field in
config.json: { "ai": { "server": "http://<this-machine>:8080" } }
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

app = FastAPI(title="SigmaServer — AI Inference")

# Runner registry: weights_name -> callable
_runners: dict[str, callable] = {}

# Model metadata registry: weights_name -> model info dict
_model_meta: dict[str, dict] = {}


def register_model(
    weights_name: str,
    *,
    id: str,
    name: str,
    description: str,
    modality: list[str],
    accepts_labels: bool = False,
):
    """Decorator that registers a model runner function with full metadata."""
    def decorator(fn):
        _runners[weights_name] = fn
        _model_meta[weights_name] = {
            "id": id,
            "name": name,
            "description": description,
            "modality": modality,
            "endpoint": "/predict",
            "weights": weights_name,
            "accepts_labels": accepts_labels,
            "labels": [],
        }
        return fn
    return decorator


# ---------------------------------------------------------------------------
# TotalSegmentator
# ---------------------------------------------------------------------------

@register_model(
    "totalsegmentator_v2",
    id="totalsegmentator",
    name="TotalSegmentator",
    description="104-structure CT segmentation (fast mode)",
    modality=["CT"],
    accepts_labels=False,
)
def run_totalsegmentator(input_path: Path, output_dir: Path, labels_path: Path | None):
    """Run TotalSegmentator on input NIfTI, return combined mask + label map."""
    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(input_path, output_dir, fast=True)

    mask_files = sorted(output_dir.glob("*.nii.gz"))
    if not mask_files:
        raise RuntimeError("TotalSegmentator produced no output files")

    ref_img = nib.load(str(mask_files[0]))
    shape = ref_img.shape[:3]
    affine = ref_img.affine
    combined = np.zeros(shape, dtype=np.uint8)
    labels = []

    STRUCTURE_COLORS = {
        "spleen": "#8b0000",
        "kidney_right": "#2e8b57",
        "kidney_left": "#228b22",
        "liver": "#daa520",
        "stomach": "#ff69b4",
        "aorta": "#ff0000",
        "pancreas": "#ffa500",
        "lung_upper_lobe_left": "#4682b4",
        "lung_lower_lobe_left": "#5f9ea0",
        "lung_upper_lobe_right": "#6495ed",
        "lung_middle_lobe_right": "#7b68ee",
        "lung_lower_lobe_right": "#87ceeb",
        "heart": "#dc143c",
        "gallbladder": "#32cd32",
        "esophagus": "#ff8c00",
        "trachea": "#00ced1",
        "small_bowel": "#f0e68c",
        "colon": "#bdb76b",
        "urinary_bladder": "#9370db",
    }

    for i, mask_file in enumerate(mask_files):
        label_val = i + 1
        if label_val > 255:
            break

        structure_name = mask_file.stem.replace(".nii", "")
        img = nib.load(str(mask_file))
        data = np.asarray(img.dataobj)
        combined[data > 0] = label_val
        color = STRUCTURE_COLORS.get(structure_name, _default_color(label_val))
        labels.append({"value": label_val, "name": structure_name, "color": color})

    combined_path = output_dir / "combined_mask.nii.gz"
    combined_img = nib.Nifti1Image(combined, affine)
    combined_img.set_data_dtype(np.uint8)
    nib.save(combined_img, str(combined_path))

    return combined_path, labels


# ---------------------------------------------------------------------------
# mhub Docker runner (generic template for containerised models)
# ---------------------------------------------------------------------------

@register_model(
    "mhub_docker",
    id="mhub-docker",
    name="mhub Docker",
    description="Generic mhub.ai Docker container runner",
    modality=[],
    accepts_labels=False,
)
def run_mhub_docker(
    input_path: Path,
    output_dir: Path,
    labels_path: Path | None,
    docker_image: str = "mhubai/totalsegmentator",
):
    """Run an mhub.ai Docker container for inference."""
    import subprocess

    input_mount = input_path.parent
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{input_mount}:/input:ro",
        "-v", f"{output_dir}:/output",
        docker_image,
        "--input", f"/input/{input_path.name}",
        "--output", "/output/output.nii.gz",
    ]

    if labels_path:
        labels_mount = labels_path.parent
        cmd.extend(["-v", f"{labels_mount}:/labels:ro"])
        cmd.extend(["--labels", f"/labels/{labels_path.name}"])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Docker failed: {result.stderr[:500]}")

    output_mask = output_dir / "output.nii.gz"
    if not output_mask.exists():
        niftis = list(output_dir.glob("*.nii.gz"))
        if niftis:
            output_mask = niftis[0]
        else:
            raise RuntimeError("Docker container produced no output")

    return output_mask, []


# ---------------------------------------------------------------------------
# Refine segmentation (placeholder — pass-through until real model added)
# ---------------------------------------------------------------------------

@register_model(
    "refine_v1",
    id="refine-seg",
    name="Refine Segmentation",
    description="Refines existing label boundaries using image features",
    modality=[],
    accepts_labels=True,
)
def run_refine(input_path: Path, output_dir: Path, labels_path: Path | None):
    """Placeholder: refine existing segmentation. Replace with real model."""
    if not labels_path or not labels_path.exists():
        raise RuntimeError("Refine model requires existing labels as input")

    output_path = output_dir / "refined.nii.gz"
    shutil.copy2(labels_path, output_path)
    return output_path, []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/models")
async def list_models():
    """Return the list of available AI models with their metadata.

    SIGMA calls this on startup and when the user opens the AI panel.
    """
    return list(_model_meta.values())


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    weights: str = Form(""),
    labels: UploadFile | None = File(None),
):
    """Run model inference on a NIfTI volume.

    Args:
        image:   Input NIfTI volume (.nii.gz)
        weights: Model weights identifier (must match a registered runner)
        labels:  Optional existing segmentation NIfTI (for accepts_labels models)

    Returns:
        NIfTI mask as binary response.
        X-AI-Labels header: JSON-encoded label array.
    """
    if weights not in _runners:
        available = list(_runners.keys())
        return Response(
            content=json.dumps({"error": f"Unknown weights '{weights}'. Available: {available}"}),
            status_code=400,
            media_type="application/json",
        )

    with tempfile.TemporaryDirectory(prefix="sigma_ai_") as tmpdir:
        tmpdir = Path(tmpdir)

        input_path = tmpdir / "input.nii.gz"
        input_path.write_bytes(await image.read())

        labels_path = None
        if labels:
            labels_path = tmpdir / "labels.nii.gz"
            labels_path.write_bytes(await labels.read())

        output_dir = tmpdir / "output"
        output_dir.mkdir()

        try:
            mask_path, label_map = _runners[weights](input_path, output_dir, labels_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json",
            )

        mask_bytes = mask_path.read_bytes()
        headers = {"X-AI-Labels": json.dumps(label_map)}

        return Response(
            content=mask_bytes,
            media_type="application/gzip",
            headers=headers,
        )


@app.get("/health")
async def health():
    """Health check — reports GPU status and available models."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "none"
        gpu_mem = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            if gpu_available else "n/a"
        )
    except ImportError:
        gpu_available = False
        gpu_name = "none"
        gpu_mem = "n/a"

    return {
        "status": "ok",
        "gpu": gpu_name,
        "gpu_memory": gpu_mem,
        "models": [m["id"] for m in _model_meta.values()],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_color(idx: int) -> str:
    colors = [
        "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff", "#ff00ff",
        "#ff8800", "#8800ff", "#00ff88", "#ff0088", "#0088ff", "#88ff00",
        "#cc4444", "#44cc44", "#4444cc", "#cccc44", "#44cccc", "#cc44cc",
    ]
    return colors[idx % len(colors)]


def main():
    parser = argparse.ArgumentParser(description="SigmaServer — AI Inference")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Registered models: {[m['id'] for m in _model_meta.values()]}")
    print(f"Starting SigmaServer on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
