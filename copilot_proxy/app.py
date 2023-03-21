import logging
import os

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from config.log_config import uvicorn_logger
from models import OpenAIinput
from utils.codegen import CodeGenProxy
from utils.errors import FauxPilotException

logging.config.dictConfig(uvicorn_logger)

codegen = CodeGenProxy(
    host=os.environ.get("TRITON_HOST", "triton"),
    port=os.environ.get("TRITON_PORT", 8001),
    verbose=os.environ.get("TRITON_VERBOSITY", False)
)

app = FastAPI(
    title="FauxPilot",
    description="This is an attempt to build a locally hosted version of GitHub Copilot. It uses the SalesForce CodeGen"
                "models inside of NVIDIA's Triton Inference Server with the FasterTransformer backend.",
    docs_url="/",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)


@app.exception_handler(FauxPilotException)
async def fauxpilot_handler(request: Request, exc: FauxPilotException):
    return JSONResponse(
        status_code=400,
        content=exc.json()
    )


# Used to support copilot.vim
@app.get("/copilot_internal/v2/token")
def get_copilot_token():
    content = {'token': '1', 'expires_at': 2600000000, 'refresh_in': 900}
    return JSONResponse(
        status_code=200,
        content=content
    )


@app.post("/v1/engines/codegen/completions")
# Used to support copilot.vim
@app.post("/v1/engines/copilot-codex/completions")
@app.post("/v1/completions")
async def completions(data: OpenAIinput):
    data = data.dict()
    try:
        content = codegen(data=data)
    except codegen.TokensExceedsMaximum as E:
        raise FauxPilotException(
            message=str(E),
            error_type="invalid_request_error",
            param=None,
            code=None,
        )

    if data.get("stream") is not None:
        return EventSourceResponse(
            content=content,
            status_code=200,
            media_type="text/event-stream"
        )
    else:
        return Response(
            status_code=200,
            content=content,
            media_type="application/json"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
