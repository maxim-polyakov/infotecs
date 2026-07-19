from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ueba_server.controller import (
    detection_controller,
    job_controller,
    model_controller,
    report_controller,
    system_controller,
    telemetry_controller,
    training_controller,
)


def create_app() -> FastAPI:
    app = FastAPI(title="UEBA Prototype API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system_controller.router)
    app.include_router(model_controller.router)
    app.include_router(training_controller.router)
    app.include_router(telemetry_controller.router)
    app.include_router(detection_controller.router)
    app.include_router(report_controller.router)
    app.include_router(job_controller.router)
    return app
