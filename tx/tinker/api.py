from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, Any, Dict, List
from uuid import uuid4

app = FastAPI(title="Tinker API Mock", version="1.0.0")

# In-memory storage for models
models_db: Dict[str, Dict[str, Any]] = {}
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

    # Store the future result
    futures_db[request_id] = model_data

    return CreateModelResponse(
        model_id=model_id,
        base_model=request.base_model,
        lora_config=request.lora_config,
        status="created",
        request_id=request_id
    )


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
            "models": ["/api/v1/create_model", "/api/v1/get_info"],
            "training": ["/api/v1/forward_backward", "/api/v1/optim_step"],
            "futures": ["/api/v1/retrieve_future"],
            "service": ["/api/v1/get_server_capabilities"],
            "telemetry": ["/api/v1/telemetry"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
