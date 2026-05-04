# SigmaServer

AI inference server for the [SIGMA](../Sigma) medical image viewer.

Runs on a GPU machine (DGX Spark, cloud GPU, or local GPU) and exposes AI segmentation models over HTTP. SIGMA discovers available tools by calling `GET /models` and sends image data to `POST /predict`.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/models` | List available AI models with metadata |
| POST | `/predict` | Run inference on a NIfTI volume |
| GET | `/health` | Health check + GPU info |

## Setup

```bash
# Clone to the GPU machine
git clone <repo-url> SigmaServer
cd SigmaServer

# Create environment and install (requires uv)
uv venv && uv sync

# Start server
uv run python server.py --host 0.0.0.0 --port 8080
```

## Connect SIGMA

In SIGMA's Preferences, set the AI server URL to `http://<this-machine>:8080`. SIGMA will call `/models` to populate the AI tools panel automatically.

Alternatively, edit `config.json` in the SIGMA directory:
```json
{
  "ai": {
    "server": "http://<this-machine>:8080"
  }
}
```

## Verify

```bash
curl http://localhost:8080/health
# {"status":"ok","gpu":"NVIDIA ...","gpu_memory":"80.0 GB","models":["totalsegmentator","refine-seg"]}

curl http://localhost:8080/models
# [{"id":"totalsegmentator","name":"TotalSegmentator",...}, ...]
```

## Adding new models

Register a new runner in `server.py` using `@register_model`:

```python
@register_model(
    "my_weights",
    id="my-model",
    name="My Model",
    description="What this model does",
    modality=["CT"],
    accepts_labels=False,
)
def run_my_model(input_path: Path, output_dir: Path, labels_path: Path | None):
    # Run inference, write output NIfTI to output_dir
    mask_path = output_dir / "output.nii.gz"
    labels = [{"value": 1, "name": "tumor", "color": "#ff0000"}]
    return mask_path, labels
```

The model appears automatically in SIGMA's AI tools panel via the `/models` endpoint.
