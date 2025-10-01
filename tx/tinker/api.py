from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, Any, Dict, List
from uuid import uuid4

app = FastAPI(title="Tinker API Mock", version="1.0.0")

# In-memory storage for models
models_db: Dict[str, Dict[str, Any]] = {}
checkpoints_db: Dict[str, List[Dict[str, Any]]] = {}
futures_db: Dict[str, Dict[str, Any]] = {}


# Pydantic Models
class LoRAConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 16
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.05


class CreateModelRequest(BaseModel):
    base_model: str
    lora_config: Optional[LoRAConfig] = None
    type: Optional[str] = None


class CreateModelResponse(BaseModel):
    model_id: str
    base_model: str
    lora_config: Optional[LoRAConfig] = None
    status: str = "created"
    request_id: str


class ModelData(BaseModel):
    base_model: str
    lora_config: Optional[LoRAConfig] = None
    model_name: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_id: str
    status: str
    model_data: ModelData


class UnloadModelResponse(BaseModel):
    model_id: str
    status: str = "unloaded"


class ForwardInput(BaseModel):
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]] = None


class ForwardBackwardInput(BaseModel):
    model_id: str
    forward_backward_input: Dict[str, Any]


class AdamParams(BaseModel):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


class OptimStepRequest(BaseModel):
    model_id: str
    adam_params: AdamParams
    type: Optional[str] = None


class SamplingParams(BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 100
    stop: Optional[List[str]] = None


class SampleRequest(BaseModel):
    prompt: str
    sampling_params: SamplingParams
    num_samples: int = 1
    base_model: Optional[str] = None
    model_path: Optional[str] = None


class SampleResponse(BaseModel):
    samples: List[str]
    prompt: str


class SaveWeightsRequest(BaseModel):
    model_id: str
    path: Optional[str] = None
    type: Optional[str] = None


class SaveWeightsResponse(BaseModel):
    model_id: str
    path: str
    status: str = "saved"


class LoadWeightsRequest(BaseModel):
    model_id: str
    path: str
    type: Optional[str] = None


class LoadWeightsResponse(BaseModel):
    model_id: str
    path: str
    status: str = "loaded"


class Checkpoint(BaseModel):
    checkpoint_id: str
    model_id: str
    path: str
    created_at: str


class FutureResponse(BaseModel):
    future_id: str
    status: str = "pending"
    request_id: Optional[str] = None


class TelemetryEvent(BaseModel):
    event: str
    event_id: str
    event_session_index: int
    severity: str
    timestamp: str
    properties: Optional[Dict[str, Any]] = None


class TelemetryRequest(BaseModel):
    events: List[TelemetryEvent]
    platform: str
    sdk_version: str
    session_id: str


class TelemetryResponse(BaseModel):
    status: Literal["accepted"] = "accepted"


class SupportedModel(BaseModel):
    model_name: Optional[str] = None


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: List[SupportedModel]


# Models endpoints
@app.post("/api/v1/create_model", response_model=CreateModelResponse)
async def create_model(request: CreateModelRequest):
    """Create a new model, optionally with a LoRA adapter."""
    model_id = f"model_{uuid4().hex[:8]}"
    request_id = f"req_{uuid4().hex[:8]}"

    model_data = {
        "model_id": model_id,
        "base_model": request.base_model,
        "lora_config": request.lora_config,
        "status": "created",
        "request_id": request_id
    }

    models_db[model_id] = model_data
    checkpoints_db[model_id] = []

    # Store the future result
    futures_db[request_id] = model_data

    return CreateModelResponse(**model_data)


class GetInfoRequest(BaseModel):
    model_id: str
    type: Optional[str] = None


@app.post("/api/v1/get_info", response_model=ModelInfoResponse)
async def get_model_info(request: GetInfoRequest):
    """Retrieve information about the current model."""
    if request.model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    model = models_db[request.model_id]
    model_data = ModelData(
        base_model=model["base_model"],
        lora_config=model["lora_config"],
        model_name=model["base_model"]
    )

    return ModelInfoResponse(
        model_id=model["model_id"],
        status=model["status"],
        model_data=model_data
    )


@app.post("/api/v1/unload_model", response_model=UnloadModelResponse)
async def unload_model(model_id: str, type: Optional[str] = None):
    """Unload model weights and end user session."""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    models_db[model_id]["status"] = "unloaded"

    return UnloadModelResponse(model_id=model_id, status="unloaded")


# Training endpoints
@app.post("/api/v1/forward", response_model=FutureResponse)
async def forward(model_id: str, forward_input: ForwardInput):
    """Perform forward pass through the model."""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    future_id = f"future_{uuid4().hex[:8]}"
    return FutureResponse(future_id=future_id, status="completed")


@app.post("/api/v1/forward_backward", response_model=FutureResponse)
async def forward_backward(request: ForwardBackwardInput):
    """Compute and accumulate gradients."""
    if request.model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    future_id = f"future_{uuid4().hex[:8]}"
    request_id = f"req_{uuid4().hex[:8]}"

    # Store the future result with required fields
    # Each loss_fn_output should contain a TensorData object
    futures_db[request_id] = {
        "loss_fn_output_type": "scalar",
        "loss_fn_outputs": [{
            "loss": {
                "data": [0.5],
                "dtype": "float32",
                "shape": [1]
            }
        }],
        "metrics": {}
    }

    return FutureResponse(future_id=future_id, status="completed", request_id=request_id)


@app.post("/api/v1/optim_step", response_model=FutureResponse)
async def optim_step(request: OptimStepRequest):
    """Update model using accumulated gradients."""
    if request.model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    future_id = f"future_{uuid4().hex[:8]}"
    request_id = f"req_{uuid4().hex[:8]}"

    # Store empty result for optim_step
    futures_db[request_id] = {}

    return FutureResponse(future_id=future_id, status="completed", request_id=request_id)


# Sampling endpoints
@app.post("/api/v1/sample", response_model=SampleResponse)
async def sample(request: SampleRequest):
    """Generate outputs from the model."""
    samples = [
        f"Generated sample {i+1} for prompt: {request.prompt}"
        for i in range(request.num_samples)
    ]

    return SampleResponse(samples=samples, prompt=request.prompt)


@app.post("/api/v1/asample", response_model=FutureResponse)
async def asample(request: SampleRequest):
    """Async generate outputs from the model."""
    future_id = f"future_{uuid4().hex[:8]}"
    return FutureResponse(future_id=future_id, status="pending")


# Weights endpoints
@app.post("/api/v1/save_weights", response_model=SaveWeightsResponse)
async def save_weights(request: SaveWeightsRequest):
    """Save current model weights to disk."""
    if request.model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    path = request.path or f"/checkpoints/{request.model_id}/weights.safetensors"

    # Add checkpoint to the list
    checkpoint = {
        "checkpoint_id": f"ckpt_{uuid4().hex[:8]}",
        "model_id": request.model_id,
        "path": path,
        "created_at": "2025-10-01T00:00:00Z"
    }
    checkpoints_db[request.model_id].append(checkpoint)

    return SaveWeightsResponse(model_id=request.model_id, path=path, status="saved")


@app.post("/api/v1/load_weights", response_model=LoadWeightsResponse)
async def load_weights(request: LoadWeightsRequest):
    """Load model weights from disk."""
    if request.model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    return LoadWeightsResponse(model_id=request.model_id, path=request.path, status="loaded")


@app.post("/api/v1/save_weights_for_sampler", response_model=SaveWeightsResponse)
async def save_weights_for_sampler(request: SaveWeightsRequest):
    """Save weights compatible with sampling/inference servers."""
    if request.model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    path = request.path or f"/checkpoints/{request.model_id}/sampler_weights.safetensors"

    return SaveWeightsResponse(model_id=request.model_id, path=path, status="saved")


@app.get("/api/v1/models/{model_id}/checkpoints", response_model=List[Checkpoint])
async def list_checkpoints(model_id: str):
    """List available model checkpoints."""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    return [Checkpoint(**ckpt) for ckpt in checkpoints_db.get(model_id, [])]


@app.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}")
async def delete_checkpoint(model_id: str, checkpoint_id: str):
    """Delete a checkpoint for a specific training run."""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    checkpoints = checkpoints_db.get(model_id, [])
    original_len = len(checkpoints)
    checkpoints_db[model_id] = [
        ckpt for ckpt in checkpoints if ckpt["checkpoint_id"] != checkpoint_id
    ]

    if len(checkpoints_db[model_id]) == original_len:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    return {"status": "deleted", "checkpoint_id": checkpoint_id}


# Service endpoints
@app.get("/api/v1/get_server_capabilities", response_model=GetServerCapabilitiesResponse)
async def get_server_capabilities():
    """Retrieve information about supported models and server capabilities."""
    # Return a list of commonly supported models
    supported_models = [
        SupportedModel(model_name="Qwen/Qwen3-8B"),
    ]
    return GetServerCapabilitiesResponse(supported_models=supported_models)


class RetrieveFutureRequest(BaseModel):
    request_id: str


# Futures endpoint
@app.post("/api/v1/retrieve_future")
async def retrieve_future(request: RetrieveFutureRequest):
    """Retrieve the result of an async operation."""
    if request.request_id not in futures_db:
        raise HTTPException(status_code=404, detail="Future not found")

    return futures_db[request.request_id]


# Telemetry endpoint
@app.post("/api/v1/telemetry", response_model=TelemetryResponse)
async def send_telemetry(request: TelemetryRequest):
    """Accept batches of SDK telemetry events for analytics and diagnostics."""
    # Just acknowledge receipt without doing anything
    return TelemetryResponse(status="accepted")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Tinker API Mock",
        "version": "1.0.0",
        "endpoints": {
            "models": ["/api/v1/create_model", "/api/v1/get_info", "/api/v1/unload_model"],
            "training": ["/api/v1/forward", "/api/v1/forward_backward", "/api/v1/optim_step"],
            "sampling": ["/api/v1/sample", "/api/v1/asample"],
            "weights": [
                "/api/v1/save_weights",
                "/api/v1/load_weights",
                "/api/v1/save_weights_for_sampler",
                "/api/v1/models/{model_id}/checkpoints"
            ],
            "telemetry": ["/api/v1/telemetry"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
