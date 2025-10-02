from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Any
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
import json
import asyncio
import subprocess

from tx.tinker.models import ModelDB, FutureDB, DB_PATH

app = FastAPI(title="Tinker API Mock", version="0.0.1")

# SQLite database path
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)

# Background engine process
background_engine_process = None


async def init_db():
    """Initialize the SQLite database with required tables."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


@app.on_event("startup")
async def startup():
    """Initialize database and start background engine on startup."""
    global background_engine_process

    await init_db()

    # Start background engine process using uv with tinker extra
    engine_path = Path(__file__).parent / "engine.py"
    background_engine_process = subprocess.Popen(
        ["uv", "run", "--extra", "tinker", "python", str(engine_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(f"Started background engine with PID {background_engine_process.pid}")


@app.on_event("shutdown")
async def shutdown():
    """Stop background engine on shutdown."""
    global background_engine_process

    if background_engine_process:
        print(f"Stopping background engine (PID {background_engine_process.pid})")
        background_engine_process.terminate()
        try:
            background_engine_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            background_engine_process.kill()
            background_engine_process.wait()
        print("Background engine stopped")


class LoRAConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] | None = None
    lora_dropout: float = 0.05


class CreateModelRequest(BaseModel):
    base_model: str
    lora_config: LoRAConfig | None = None
    type: str | None = None


class CreateModelResponse(BaseModel):
    model_id: str
    base_model: str
    lora_config: LoRAConfig | None = None
    status: str = "created"
    request_id: str


class ModelData(BaseModel):
    base_model: str
    lora_config: LoRAConfig | None = None
    model_name: str | None = None


class ModelInfoResponse(BaseModel):
    model_id: str
    status: str
    model_data: ModelData


class ForwardBackwardInput(BaseModel):
    model_id: str
    forward_backward_input: dict[str, Any]


class AdamParams(BaseModel):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


class OptimStepRequest(BaseModel):
    model_id: str
    adam_params: AdamParams
    type: str | None = None


class FutureResponse(BaseModel):
    future_id: str
    status: str = "pending"
    request_id: str | None = None


class TelemetryEvent(BaseModel):
    event: str
    event_id: str
    event_session_index: int
    severity: str
    timestamp: str
    properties: dict[str, Any] | None = None


class TelemetryRequest(BaseModel):
    events: list[TelemetryEvent]
    platform: str
    sdk_version: str
    session_id: str


class TelemetryResponse(BaseModel):
    status: Literal["accepted"] = "accepted"


class SupportedModel(BaseModel):
    model_name: str | None = None


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: list[SupportedModel]


@app.post("/api/v1/create_model", response_model=CreateModelResponse)
async def create_model(request: CreateModelRequest):
    """Create a new model, optionally with a LoRA adapter."""
    model_id = f"model_{uuid4().hex[:8]}"
    request_id = f"req_{uuid4().hex[:8]}"

    async with AsyncSession(engine) as session:
        # Store in models table
        model_db = ModelDB(
            model_id=model_id,
            base_model=request.base_model,
            lora_config=json.dumps(request.lora_config.model_dump()) if request.lora_config else None,
            status="created",
            request_id=request_id
        )
        session.add(model_db)

        # Store in futures table - result is the same as request for create_model
        future_db = FutureDB(
            request_id=request_id,
            request_type="create_model",
            model_id=model_id,
            request_data=json.dumps(request.model_dump()),
            result_data=json.dumps({
                "model_id": model_id,
                "base_model": request.base_model,
                "lora_config": request.lora_config.model_dump() if request.lora_config else None,
                "status": "created",
                "request_id": request_id
            }),
            status="completed",
            completed_at=datetime.utcnow()
        )
        session.add(future_db)

        await session.commit()

    return CreateModelResponse(
        model_id=model_id,
        base_model=request.base_model,
        lora_config=request.lora_config,
        status="created",
        request_id=request_id
    )


class GetInfoRequest(BaseModel):
    model_id: str
    type: str | None = None


@app.post("/api/v1/get_info", response_model=ModelInfoResponse)
async def get_model_info(request: GetInfoRequest):
    """Retrieve information about the current model."""
    async with AsyncSession(engine) as session:
        statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
        result = await session.exec(statement)
        model = result.first()

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        lora_config = None
        if model.lora_config:
            lora_config = LoRAConfig(**json.loads(model.lora_config))

        model_data = ModelData(
            base_model=model.base_model,
            lora_config=lora_config,
            model_name=model.base_model
        )

        return ModelInfoResponse(
            model_id=model.model_id,
            status=model.status,
            model_data=model_data
        )


@app.post("/api/v1/forward_backward", response_model=FutureResponse)
async def forward_backward(request: ForwardBackwardInput):
    """Compute and accumulate gradients."""
    async with AsyncSession(engine) as session:
        statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
        result = await session.exec(statement)
        model = result.first()

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        request_id = f"req_{uuid4().hex[:8]}"

        # Store the request for background processing
        future_db = FutureDB(
            request_id=request_id,
            request_type="forward_backward",
            model_id=request.model_id,
            request_data=json.dumps(request.model_dump()),
            result_data=None,  # Will be filled by background worker
            status="pending"
        )
        session.add(future_db)
        await session.commit()

        return FutureResponse(future_id=request_id, status="pending", request_id=request_id)


@app.post("/api/v1/optim_step", response_model=FutureResponse)
async def optim_step(request: OptimStepRequest):
    """Update model using accumulated gradients."""
    async with AsyncSession(engine) as session:
        statement = select(ModelDB).where(ModelDB.model_id == request.model_id)
        result = await session.exec(statement)
        model = result.first()

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        request_id = f"req_{uuid4().hex[:8]}"

        # Store the request for background processing
        future_db = FutureDB(
            request_id=request_id,
            request_type="optim_step",
            model_id=request.model_id,
            request_data=json.dumps(request.model_dump()),
            result_data=None,  # Will be filled by background worker
            status="pending"
        )
        session.add(future_db)
        await session.commit()

        return FutureResponse(future_id=request_id, status="pending", request_id=request_id)


@app.get("/api/v1/get_server_capabilities", response_model=GetServerCapabilitiesResponse)
async def get_server_capabilities():
    """Retrieve information about supported models and server capabilities."""
    supported_models = [
        SupportedModel(model_name="Qwen/Qwen3-8B"),
    ]
    return GetServerCapabilitiesResponse(supported_models=supported_models)


class RetrieveFutureRequest(BaseModel):
    request_id: str


@app.post("/api/v1/retrieve_future")
async def retrieve_future(request: RetrieveFutureRequest):
    """Retrieve the result of an async operation, waiting until it's available."""
    timeout = 300  # 5 minutes
    poll_interval = 0.1  # 100ms

    for _ in range(int(timeout / poll_interval)):
        async with AsyncSession(engine) as session:
            statement = select(FutureDB).where(FutureDB.request_id == request.request_id)
            result = await session.exec(statement)
            future = result.first()

            if not future:
                raise HTTPException(status_code=404, detail="Future not found")

            if future.status == "completed" and future.result_data:
                return json.loads(future.result_data)

            if future.status == "failed":
                error = json.loads(future.result_data).get("error", "Unknown error") if future.result_data else "Unknown error"
                raise HTTPException(status_code=500, detail=error)

        await asyncio.sleep(poll_interval)

    raise HTTPException(status_code=408, detail="Timeout waiting for result")


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
        "version": "0.0.1",
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
