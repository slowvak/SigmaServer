"""SigmaServer — AI inference server for the SIGMA medical image viewer.

Runs on a GPU machine (DGX Spark, cloud GPU instance, or local GPU).
Loads models once at startup and exposes them via HTTP API.

Endpoints:
    GET  /models         — list available AI tools with metadata
    POST /models/upload  — upload and register a custom model file
    POST /models/reload  — reload models_registry.json without restart
    POST /predict        — run model inference on a NIfTI volume
    GET  /health         — health check + GPU info

Usage:
    python server.py [--host 0.0.0.0] [--port 8050]

SIGMA image server discovers this server via the "ai.server" field in
config.json: { "ai": { "server": "http://<this-machine>:8050" } }
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

REGISTRY_FILE = Path(__file__).parent / "models_registry.json"

# ---------------------------------------------------------------------------
# Live inference progress — patched directly into tqdm.update at startup
# ---------------------------------------------------------------------------

_live_progress: dict = {"pct": 0, "phase": "idle"}


def _install_tqdm_hook() -> None:
    """Patch tqdm.tqdm.update so every progress step updates _live_progress.

    Patching the method on the class (rather than replacing the class or
    redirecting stderr) works regardless of how each module imported tqdm.
    """
    try:
        import tqdm as _tqdm_mod
        _orig = _tqdm_mod.tqdm.update

        def _update(self, n=1):
            result = _orig(self, n)
            total = getattr(self, "total", None)
            if total:
                _live_progress["pct"] = min(99, int(100 * self.n / total))
                _live_progress["phase"] = "running"
            return result

        _tqdm_mod.tqdm.update = _update
        print("[SigmaServer] tqdm progress hook installed")
    except Exception as exc:
        print(f"[SigmaServer] Could not install tqdm hook: {exc}")
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


def _infer_monai_unet_arch(state_dict: dict) -> dict | None:
    """Read weight shapes from a MONAI UNet state dict to reconstruct its architecture.

    Handles Lightning checkpoints (nested under 'state_dict' key), DataParallel
    ('module.' prefix), and any other outer module wrapper prefix.
    """
    import re

    # Unwrap Lightning / other checkpoint wrappers that nest weights under a key
    for nest_key in ("state_dict", "model_state_dict", "model"):
        if nest_key in state_dict and isinstance(state_dict.get(nest_key), dict):
            state_dict = state_dict[nest_key]
            break

    # Find the prefix that precedes "model.{N}.conv.unit0.conv.weight"
    prefix = None
    for k in state_dict:
        m = re.match(r"^(.*?)model\.\d+\.conv\.unit0\.conv\.weight$", k)
        if m:
            prefix = m.group(1)
            break
    if prefix is None:
        return None

    # Build a prefix-stripped view
    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # Collect encoder levels: model.{N}.conv.unit0.conv.weight
    encoder: dict[int, object] = {}
    for k, v in stripped.items():
        m = re.match(r"^model\.(\d+)\.conv\.unit0\.conv\.weight$", k)
        if m:
            encoder[int(m.group(1))] = v
    if not encoder:
        return None

    sorted_idxs = sorted(encoder.keys())
    first_w = encoder[sorted_idxs[0]]

    # spatial_dims from conv weight rank: shape is [out, in, *kernel]
    spatial_dims = len(first_w.shape) - 2
    if spatial_dims not in (2, 3):
        return None

    in_channels  = int(first_w.shape[1])
    channels     = [int(encoder[i].shape[0]) for i in sorted_idxs]
    strides      = [2] * (len(channels) - 1)

    # out_channels from the last model level's first conv output
    all_idxs = set()
    for k in stripped:
        m = re.match(r"^model\.(\d+)\.", k)
        if m:
            all_idxs.add(int(m.group(1)))
    last_idx  = max(all_idxs)
    last_unit = stripped.get(f"model.{last_idx}.conv.unit0.conv.weight")
    out_channels = int(last_unit.shape[0]) if last_unit is not None else 2

    # num_res_units from how many unit{N} keys exist per encoder level
    max_unit_idx = 0
    for k in stripped:
        m = re.match(r"^model\.\d+\.conv\.unit(\d+)\.conv\.weight$", k)
        if m:
            max_unit_idx = max(max_unit_idx, int(m.group(1)))
    num_res_units = max_unit_idx

    arch = {
        "format": "monai_unet",
        "spatial_dims": spatial_dims,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "channels": channels,
        "strides": strides,
        "num_res_units": num_res_units,
        "_key_prefix": prefix,  # stored so load_state_dict can strip it
    }
    print(f"[upload] Auto-detected MONAI UNet arch from state dict: {arch}")
    return arch


def _make_torch_runner(model_path: str, model_info: dict):
    import torch

    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(obj, torch.nn.Module):
        model = obj.to(device).eval()
    elif isinstance(obj, dict):
        inferred = _infer_monai_unet_arch(obj)
        if inferred:
            return _make_monai_unet_runner(model_path, inferred)
        sample_keys = list(obj.keys())[:8]
        print(f"[upload] Unrecognized checkpoint. Top-level keys: {sample_keys}")
        raise RuntimeError(
            f"Unrecognized checkpoint format. Top-level keys: {sample_keys}. "
            "Supported auto-detection: MONAI UNet state dicts and Lightning checkpoints."
        )
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(obj)}")

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
    if isinstance(sd, dict):
        for nest_key in ("state_dict", "model_state_dict", "model"):
            if nest_key in sd and isinstance(sd.get(nest_key), dict):
                sd = sd[nest_key]
                break
        prefix = model_info.get("_key_prefix", "")
        if prefix:
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
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


# ---------------------------------------------------------------------------
# Built-in models (always available, no user registration required)
# ---------------------------------------------------------------------------

def _totalsegmentator_device() -> tuple[str, bool]:
    """Return (device, use_fast) suited to the current platform.

    macOS: always CPU + fast=True. MPS (Apple Silicon GPU) is NOT used because
    nnU-Net has incomplete MPS support — unsupported ops fall back silently and
    NaN values propagate through the network, producing an all-zero output mask.
    Other: gpu when CUDA is available, else CPU; fast=False (full 1.5 mm model).
    """
    if sys.platform == "darwin":
        return "cpu", True
    import torch
    device = "gpu" if torch.cuda.is_available() else "cpu"
    return device, False


def _label_color(idx: int) -> dict:
    import colorsys
    hue = (idx * 0.618033988749895) % 1.0  # golden-ratio spread
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.90)
    return {"r": int(r * 255), "g": int(g * 255), "b": int(b * 255)}


def _get_totalsegmentator_labels(task: str) -> list[dict]:
    """Return [{value, name, color}] label objects for each non-background class."""
    try:
        from totalsegmentator.map_to_binary import class_map
        print(f"[TotalSeg] class_map keys: {list(class_map.keys())}")
        mapping: dict[int, str] = class_map.get(task, {})
        print(f"[TotalSeg] task='{task}' → {len(mapping)} label entries")
        if not mapping:
            return []
        return [
            {"value": idx, "name": name, "color": _label_color(idx)}
            for idx, name in sorted(mapping.items())
        ]
    except Exception as e:
        print(f"[TotalSeg] _get_totalsegmentator_labels error: {e}")
        return []


def _make_totalsegmentator_runner(task: str):
    from totalsegmentator.python_api import totalsegmentator as _ts
    device, platform_fast = _totalsegmentator_device()

    def run(input_path: Path, output_dir: Path, labels_path: Path | None, fast: bool | None = None):
        use_fast = platform_fast if fast is None else fast
        input_img = nib.load(str(input_path))
        in_data = np.asarray(input_img.dataobj)
        print(f"[TotalSeg] Input shape={input_img.shape} dtype={in_data.dtype} "
              f"range=[{in_data.min():.0f}, {in_data.max():.0f}] non-zero={np.count_nonzero(in_data)}"
              f" fast={use_fast}")

        _live_progress["pct"] = 0
        _live_progress["phase"] = "running"
        output_img = _ts(input_img, task=task, ml=True, verbose=True,
                         device=device, fast=use_fast)

        if output_img is None:
            print("[TotalSeg] ERROR: _ts returned None")
        else:
            out_data = np.asarray(output_img.dataobj)
            print(f"[TotalSeg] Output shape={output_img.shape} dtype={out_data.dtype} "
                  f"non-zero={np.count_nonzero(out_data)} unique={np.unique(out_data).tolist()[:10]}")
        _live_progress["pct"] = 100
        _live_progress["phase"] = "done"
        out_path = output_dir / "prediction.nii.gz"
        nib.save(output_img, str(out_path))
        return out_path, _get_totalsegmentator_labels(task)

    return run


_BUILTIN_MODELS = [
    {
        "id": "totalsegmentator",
        "name": "TotalSegmentator",
        "description": "Segmentation of 117 anatomical structures in CT images.",
        "modality": ["CT"],
        "task": "total",
    },
    {
        "id": "totalsegmentator_mr",
        "name": "TotalSegmentator MR",
        "description": "Segmentation of 50 anatomical structures in MR images.",
        "modality": ["MR"],
        "task": "total_mr",
    },
]


def load_builtin_models() -> None:
    """Register built-in runners that are always available regardless of the registry."""
    for spec in _BUILTIN_MODELS:
        model_id = spec["id"]
        try:
            runner = _make_totalsegmentator_runner(spec["task"])
        except Exception as e:
            print(f"[builtin] {model_id} unavailable: {e}")
            continue
        _runners[model_id] = runner
        _model_meta[model_id] = {
            "id": model_id,
            "name": spec["name"],
            "description": spec["description"],
            "modality": spec["modality"],
            "endpoint": "/predict",
            "weights": model_id,
            "accepts_labels": False,
            "labels": [],
        }
        print(f"[builtin] Loaded '{model_id}'")


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

@app.get("/progress")
async def get_progress():
    """Return the current inference progress as {pct: 0-100, phase: str}."""
    return _live_progress


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
    fast: str = Form(""),
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
            fast_flag = True if fast == "true" else (False if fast == "false" else None)
            mask_path, label_map = await asyncio.to_thread(
                _runners[weights], input_path, output_dir, labels_path, fast=fast_flag
            )
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


@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    config: UploadFile | None = File(None),
    name: str = Form(None),
    description: str = Form(""),
    arch: str = Form(""),
):
    """Upload and register a custom model.

    Accepted formats: .pth/.pt, .onnx, .safetensors.
    - .safetensors requires a HuggingFace config.json alongside it.
    - .pth/.pt state dicts require a model config JSON with a "format" key
      (e.g. {"format":"monai_unet","spatial_dims":3,"in_channels":1,"out_channels":2,
              "channels":[16,32,64,128,256],"strides":[2,2,2,2]}).
      Full nn.Module checkpoints load without a config.
    """
    _EXT_TO_FMT = {".onnx": "onnx", ".pth": "torch", ".pt": "torch", ".safetensors": "safetensors"}

    suffix = Path(file.filename).suffix.lower()
    fmt = _EXT_TO_FMT.get(suffix)
    if not fmt:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Accepted: .pth, .pt, .onnx, .safetensors",
        )

    if fmt == "safetensors" and config is None:
        raise HTTPException(status_code=400, detail="SafeTensors models require a config.json")

    stem = Path(file.filename).stem
    model_id = stem.replace(" ", "-").lower()
    display_name = name or stem

    models_dir = Path(__file__).parent / "models" / model_id
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / file.filename
    model_path.write_bytes(await file.read())

    paths: dict[str, str] = (
        {file.filename: str(model_path)} if fmt == "safetensors" else {"model": str(model_path)}
    )

    model_info: dict = {}
    if config is not None:
        config_bytes = await config.read()
        config_path = models_dir / "config.json"
        config_path.write_bytes(config_bytes)
        paths["config.json"] = str(config_path)

        # For torch state dicts, the JSON may specify architecture via "format"
        if fmt == "torch":
            try:
                arch_dict = json.loads(config_bytes)
                arch_fmt = arch_dict.get("format")
                if arch_fmt in _RUNNER_FACTORIES:
                    fmt = arch_fmt
                    model_info = arch_dict
            except Exception:
                pass

    # Also accept architecture spec from the arch form field (no separate file needed)
    if fmt == "torch" and arch:
        try:
            arch_dict = json.loads(arch)
            arch_fmt = arch_dict.get("format")
            if arch_fmt in _RUNNER_FACTORIES:
                fmt = arch_fmt
                model_info = arch_dict
        except Exception:
            pass

    entry = {
        "id": model_id,
        "name": display_name,
        "description": description,
        "modality": [],
        "format": fmt,
        "paths": paths,
        "accepts_labels": False,
        "model_info": model_info,
    }

    registry_data = json.loads(REGISTRY_FILE.read_text()) if REGISTRY_FILE.exists() else {"models": []}
    registry_data["models"] = [m for m in registry_data["models"] if m["id"] != model_id]
    registry_data["models"].append(entry)
    REGISTRY_FILE.write_text(json.dumps(registry_data, indent=2) + "\n")

    factory = _RUNNER_FACTORIES[fmt]
    try:
        if fmt == "safetensors":
            runner = await asyncio.to_thread(factory, paths, model_info)
        else:
            runner = await asyncio.to_thread(factory, paths["model"], model_info)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Model failed to load: {exc}")

    _runners[model_id] = runner
    meta = {
        "id": model_id,
        "name": display_name,
        "description": description,
        "modality": [],
        "endpoint": "/predict",
        "weights": model_id,
        "accepts_labels": False,
        "labels": [],
    }
    _model_meta[model_id] = meta
    _registry_model_ids.add(model_id)
    print(f"[upload] Registered '{display_name}' (id={model_id}, fmt={fmt})")
    return meta


@app.post("/models/reload")
async def reload_models():
    """Re-read models_registry.json and load any newly registered models."""
    load_builtin_models()
    load_registered_models()
    return {"loaded": list(_runners.keys())}


@app.get("/health")
async def health():
    """Health check — reports GPU status and available models."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        elif torch.backends.mps.is_available():
            gpu_available = True
            gpu_name = "Apple MPS"
            gpu_mem = "n/a"
        else:
            gpu_available = False
            gpu_name = "none"
            gpu_mem = "n/a"
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
    import os
    # Allow TotalSegmentator weight downloads through corporate TLS proxies (e.g. Zscaler).
    # requests/urllib3 honour REQUESTS_CA_BUNDLE; SSL_CERT_FILE is set by many corp environments.
    if ssl_cert := os.environ.get("SSL_CERT_FILE"):
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ssl_cert)

    parser = argparse.ArgumentParser(description="SigmaServer — AI Inference")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    _install_tqdm_hook()
    load_builtin_models()
    load_registered_models()
    print(f"Registered models: {[m['id'] for m in _model_meta.values()]}")
    print(f"Starting SigmaServer on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
