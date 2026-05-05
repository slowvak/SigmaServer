"""SigmaServer — AI inference server for the SIGMA medical image viewer.

Runs on a GPU machine (DGX Spark, cloud GPU instance, or local GPU).
Loads models once at startup and exposes them via HTTP API.

Endpoints:
    GET  /models         — list available AI tools with metadata
    POST /predict        — run model inference on a NIfTI volume
    GET  /health         — health check + GPU info

Usage:
    python server.py [--host 0.0.0.0] [--port 8050]

SIGMA image server discovers this server via the "ai.server" field in
config.json: { "ai": { "server": "http://<this-machine>:8050" } }
"""

import argparse
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

REGISTRY_FILE = Path(__file__).parent / "models_registry.json"
_registry_model_ids: set[str] = set()
_registry_active: bool = False  # True once registry file with entries is found

app = FastAPI(title="SigmaServer — AI Inference")

# Runner registry: weights_name -> callable
_runners: dict[str, callable] = {}

# Model metadata registry: weights_name -> model info dict
_model_meta: dict[str, dict] = {}




# ---------------------------------------------------------------------------
# Registry-based model loading (AddModel.py writes models_registry.json)
# ---------------------------------------------------------------------------

def _make_onnx_runner(model_path: str, model_info: dict):
    import onnxruntime as ort

    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    n_outputs = session.get_outputs()[0].shape[1] if session.get_outputs()[0].shape else 1

    def run(input_path: Path, output_dir: Path, labels_path: Path | None):
        img = nib.load(str(input_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        lo, hi = data.min(), data.max()
        data = (data - lo) / (hi - lo + 1e-8)
        x = data[np.newaxis, np.newaxis]  # [1, 1, D, H, W]
        raw = session.run(None, {input_name: x})[0]  # [1, C, D, H, W] or [1, 1, D, H, W]
        if raw.shape[1] > 1:
            mask = raw[0].argmax(axis=0).astype(np.uint8)
        else:
            mask = (raw[0, 0] > 0.5).astype(np.uint8)
        out = output_dir / "prediction.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine), str(out))
        return out, []

    return run


def _make_torch_runner(model_path: str, model_info: dict):
    import torch

    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(obj, torch.nn.Module):
        model = obj.to(device).eval()
    else:
        raise RuntimeError(
            f"Checkpoint at {model_path} is a state dict, not a full nn.Module. "
            "Wrap it in your model class before registering."
        )

    def run(input_path: Path, output_dir: Path, labels_path: Path | None):
        img = nib.load(str(input_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        lo, hi = data.min(), data.max()
        data = (data - lo) / (hi - lo + 1e-8)
        x = torch.from_numpy(data[np.newaxis, np.newaxis]).to(device)  # [1, 1, D, H, W]
        with torch.no_grad():
            raw = model(x)
        raw = raw.cpu().numpy()
        if raw.shape[1] > 1:
            mask = raw[0].argmax(axis=0).astype(np.uint8)
        else:
            mask = (raw[0, 0] > 0.5).astype(np.uint8)
        out = output_dir / "prediction.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine), str(out))
        return out, []

    return run


def _make_safetensors_runner(paths: dict[str, str], model_info: dict):
    if "config.json" not in paths:
        raise RuntimeError(
            "SafeTensors models require a config.json to determine the architecture. "
            "Re-register with: python AddModel.py model.safetensors config.json"
        )

    from transformers import AutoConfig, AutoModelForSemanticSegmentation

    config_dir = str(Path(paths["config.json"]).parent)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    try:
        model = AutoModelForSemanticSegmentation.from_pretrained(config_dir).to(device).eval()
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(config_dir).to(device).eval()

    import torch

    def run(input_path: Path, output_dir: Path, labels_path: Path | None):
        img = nib.load(str(input_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        lo, hi = data.min(), data.max()
        data = (data - lo) / (hi - lo + 1e-8)
        x = torch.from_numpy(data[np.newaxis, np.newaxis]).to(device)
        with torch.no_grad():
            out = model(pixel_values=x)
        logits = out.logits.cpu().numpy()
        if logits.shape[1] > 1:
            mask = logits[0].argmax(axis=0).astype(np.uint8)
        else:
            mask = (logits[0, 0] > 0.5).astype(np.uint8)
        out_path = output_dir / "prediction.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine), str(out_path))
        return out_path, []

    return run


def _make_monai_unet_runner(model_path: str, model_info: dict):
    import torch
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=model_info.get("spatial_dims", 3),
        in_channels=model_info.get("in_channels", 1),
        out_channels=model_info.get("out_channels", 2),
        channels=model_info.get("channels", (16, 32, 64, 128, 256)),
        strides=model_info.get("strides", (2, 2, 2, 2)),
        num_res_units=model_info.get("num_res_units", 2),
        norm=model_info.get("norm", "batch"),
    )
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def run(input_path: Path, output_dir: Path, labels_path: Path | None):
        import torch as _torch
        img = nib.load(str(input_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        lo, hi = data.min(), data.max()
        data = (data - lo) / (hi - lo + 1e-8)
        x = _torch.from_numpy(data[np.newaxis, np.newaxis]).to(device)
        with _torch.no_grad():
            raw = model(x)
        raw = raw.cpu().numpy()
        if raw.shape[1] > 1:
            mask = raw[0].argmax(axis=0).astype(np.uint8)
        else:
            mask = (raw[0, 0] > 0.5).astype(np.uint8)
        out = output_dir / "prediction.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine), str(out))
        return out, []

    return run


_RUNNER_FACTORIES = {
    "onnx": _make_onnx_runner,
    "torch": _make_torch_runner,
    "monai_unet": _make_monai_unet_runner,
    "safetensors": _make_safetensors_runner,
}


def load_registered_models() -> None:
    """Read models_registry.json and register each entry into _runners / _model_meta."""
    if not REGISTRY_FILE.exists():
        return

    global _registry_active
    registry = json.loads(REGISTRY_FILE.read_text())
    _registry_model_ids.clear()
    if registry.get("models"):
        _registry_active = True
    for entry in registry.get("models", []):
        model_id = entry["id"]
        fmt = entry["format"]
        paths = entry["paths"]

        factory = _RUNNER_FACTORIES.get(fmt)
        if factory is None:
            print(f"[registry] Unknown format '{fmt}' for model '{model_id}' — skipped")
            continue

        try:
            if fmt == "safetensors":
                runner = factory(paths, entry.get("model_info", {}))
            else:
                runner = factory(paths["model"], entry.get("model_info", {}))
        except Exception as exc:
            print(f"[registry] Failed to load '{model_id}': {exc}")
            continue

        _runners[model_id] = runner
        _model_meta[model_id] = {
            "id": model_id,
            "name": entry["name"],
            "description": entry["description"],
            "modality": entry["modality"],
            "endpoint": "/predict",
            "weights": model_id,
            "accepts_labels": entry.get("accepts_labels", False),
            "labels": [],
        }
        _registry_model_ids.add(model_id)
        print(f"[registry] Loaded '{model_id}' ({fmt})")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/models")
async def list_models():
    """Return the list of available AI models with their metadata.

    Returns only registry-loaded models when the registry has entries,
    so built-in placeholders don't appear alongside user-registered models.
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


@app.post("/models/reload")
async def reload_models():
    """Re-read models_registry.json and load any newly registered models."""
    load_registered_models()
    return {"loaded": list(_runners.keys())}


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



def main():
    parser = argparse.ArgumentParser(description="SigmaServer — AI Inference")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    load_registered_models()
    print(f"Registered models: {[m['id'] for m in _model_meta.values()]}")
    print(f"Starting SigmaServer on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
