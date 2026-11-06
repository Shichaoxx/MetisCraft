"""HTTP service accepting the official ``/v1/systemone`` request format."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException

from metiscraft import __version__
from metiscraft.data.jev import JevFormatError
from metiscraft.serve.predictor import Predictor


def create_app(predictor: Predictor):
    app = FastAPI(title="MetisCraft", version=__version__)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": predictor.name,
            "calibrated": predictor.calibration is not None,
            "base_verification": predictor.meta.get("base_verification", "in_memory"),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"data": [{"id": predictor.name, "object": "model", "owned_by": "metiscraft"}]}

    @app.post("/v1/systemone")
    def systemone(request: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        try:
            return predictor.answer_request(request)
        except (JevFormatError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def serve(predictor: Predictor, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    logger = logging.getLogger("metiscraft.serve.predictor")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    uvicorn.run(create_app(predictor), host=host, port=port)
