#!/usr/bin/env python3
"""AddModel — register a local model file with SigmaServer.

Detects format from file extension(s), validates the model loads cleanly,
writes an entry to models_registry.json, and optionally notifies a running
server to pick it up immediately.

Supported formats:
    .onnx                         — ONNX Runtime
    .pth / .pt                    — PyTorch checkpoint or full nn.Module
    .safetensors [+ config.json]  — HuggingFace SafeTensors / Transformers

Usage:
    python AddModel.py model.onnx
    python AddModel.py model.safetensors --name "My Seg Model" --modality CT
    python AddModel.py model.safetensors config.json --name "HF Seg" --modality CT MRI
    python AddModel.py model.pth --name "My Model" --id my-model
    python AddModel.py model.onnx --server http://localhost:8080
"""

import argparse
import json
import sys
from pathlib import Path

REGISTRY_FILE = Path(__file__).parent / "models_registry.json"

_EXT_FORMAT = {
    ".onnx": "onnx",
    ".pth": "torch",
    ".pt": "torch",
    ".safetensors": "safetensors",
}


def detect_format(paths: list[Path]) -> tuple[str, dict[str, str]]:
    """Return (format_name, file_map) where file_map maps role → absolute path."""
    by_fmt: dict[str, list[Path]] = {}
    config_path: Path | None = None

    for p in paths:
        if p.suffix == ".json" and p.stem == "config":
            config_path = p
            continue
        fmt = _EXT_FORMAT.get(p.suffix)
        if fmt is None:
            print(f"Warning: unsupported extension skipped: {p}", file=sys.stderr)
            continue
        by_fmt.setdefault(fmt, []).append(p)

    if not by_fmt:
        raise ValueError(
            "No supported model files found.\n"
            "Supported extensions: .onnx  .pth  .pt  .safetensors"
        )
    if len(by_fmt) > 1:
        raise ValueError(f"Mixed formats not supported in one call: {list(by_fmt)}")

    fmt = next(iter(by_fmt))
    model_files = by_fmt[fmt]
    file_map: dict[str, str] = {}

    if fmt in ("onnx", "torch"):
        if len(model_files) != 1:
            raise ValueError(f"Provide exactly one {model_files[0].suffix} file")
        file_map["model"] = str(model_files[0].resolve())

    elif fmt == "safetensors":
        for sf in model_files:
            file_map[sf.name] = str(sf.resolve())
        # accept explicit config.json or auto-discover from sibling directory
        cfg = config_path or model_files[0].parent / "config.json"
        if cfg.exists():
            file_map["config.json"] = str(cfg.resolve())

    return fmt, file_map


# ---------------------------------------------------------------------------
# Per-format validation (loads the model to catch bad files early)
# ---------------------------------------------------------------------------

def _validate_onnx(file_map: dict[str, str]) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "  onnxruntime not installed — skipping validation.\n"
            "  Install with: pip install onnxruntime-gpu",
            file=sys.stderr,
        )
        return {}

    session = ort.InferenceSession(
        file_map["model"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    inputs = [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()]
    outputs = [{"name": o.name, "shape": o.shape, "type": o.type} for o in session.get_outputs()]
    print(f"  Inputs:  {[i['name'] + str(i['shape']) for i in inputs]}")
    print(f"  Outputs: {[o['name'] + str(o['shape']) for o in outputs]}")
    return {"inputs": inputs, "outputs": outputs}


def _validate_torch(file_map: dict[str, str]) -> dict:
    try:
        import torch
    except ImportError:
        print("  torch not installed — skipping validation.", file=sys.stderr)
        return {}

    obj = torch.load(file_map["model"], map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        sample_keys = list(obj.keys())[:8]
        is_state = all(hasattr(v, "shape") for v in obj.values())
        print(f"  {'State dict' if is_state else 'Checkpoint'} with {len(obj)} keys. Sample: {sample_keys}")
        return {"checkpoint_type": "state_dict" if is_state else "checkpoint", "sample_keys": sample_keys}
    else:
        print(f"  Full model: {type(obj).__name__}")
        return {"checkpoint_type": "full_model", "class": type(obj).__name__}


def _validate_safetensors(file_map: dict[str, str]) -> dict:
    info: dict = {"has_config": "config.json" in file_map}
    try:
        from safetensors import safe_open
        tensor_keys: dict[str, list[str]] = {}
        for role, path in file_map.items():
            if path.endswith(".safetensors"):
                with safe_open(path, framework="pt", device="cpu") as f:
                    tensor_keys[role] = list(f.keys())[:6]
        print(f"  Tensor keys (sample): {tensor_keys}")
        info["tensor_keys_sample"] = tensor_keys
    except ImportError:
        print(
            "  safetensors not installed — skipping validation.\n"
            "  Install with: pip install safetensors",
            file=sys.stderr,
        )

    if info["has_config"]:
        with open(file_map["config.json"]) as fh:
            cfg = json.load(fh)
        info["model_type"] = cfg.get("model_type", "unknown")
        info["architectures"] = cfg.get("architectures", [])
        print(f"  Model type: {info['model_type']}, architectures: {info['architectures']}")

    return info


_VALIDATORS = {
    "onnx": _validate_onnx,
    "torch": _validate_torch,
    "safetensors": _validate_safetensors,
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {"models": []}


def save_registry(registry: dict) -> None:
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a model file with SigmaServer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="FILE",
                        help="Model file(s) to register")
    parser.add_argument("--id", dest="model_id",
                        help="Model identifier used in /predict (default: derived from filename)")
    parser.add_argument("--name",
                        help="Human-readable model name (default: filename stem)")
    parser.add_argument("--description", default="",
                        help="Short description shown in /models list")
    parser.add_argument("--modality", nargs="+", default=[], metavar="M",
                        help="Imaging modality tags, e.g. --modality CT MRI")
    parser.add_argument("--accepts-labels", action="store_true",
                        help="Model expects an existing segmentation mask as additional input")
    parser.add_argument("--server", metavar="URL",
                        help="Notify a running SigmaServer to reload immediately "
                             "(e.g. http://localhost:8080)")
    args = parser.parse_args()

    for p in args.files:
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Detecting format from: {[str(p) for p in args.files]}")
    fmt, file_map = detect_format(args.files)
    print(f"Format: {fmt}")

    primary = next(p for p in args.files if _EXT_FORMAT.get(p.suffix) == fmt)
    model_id = args.model_id or primary.stem.replace(" ", "-").lower()
    name = args.name or primary.stem

    print("Validating model...")
    model_info = _VALIDATORS[fmt](file_map)

    entry = {
        "id": model_id,
        "name": name,
        "description": args.description,
        "modality": args.modality,
        "format": fmt,
        "paths": file_map,
        "accepts_labels": args.accepts_labels,
        "model_info": model_info,
    }

    registry = load_registry()
    registry["models"] = [m for m in registry["models"] if m["id"] != model_id]
    registry["models"].append(entry)
    save_registry(registry)
    print(f"Registered '{name}' (id={model_id}) → {REGISTRY_FILE}")

    if args.server:
        try:
            import httpx
            url = args.server.rstrip("/") + "/models/reload"
            resp = httpx.post(url, timeout=10)
            if resp.status_code == 200:
                print(f"Server at {args.server} reloaded successfully.")
            else:
                print(f"Server reload returned {resp.status_code}: {resp.text}", file=sys.stderr)
        except Exception as exc:
            print(f"Could not notify server: {exc}", file=sys.stderr)

    if not args.server:
        print("Restart SigmaServer (or POST /models/reload) to activate this model.")


if __name__ == "__main__":
    main()
