import os

import uvicorn
from fastapi import FastAPI, Response
from sse_starlette.sse import EventSourceResponse

from models import OpenAIinput
from utils.codegen import CodeGenProxy

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


@app.post("/v1/engines/codegen/completions", status_code=200)
@app.post("/v1/completions", status_code=200)
async def completions(data: OpenAIinput):
    data = data.dict()
    print(data)
    if data.get("stream") is not None:
        return EventSourceResponse(
            content=codegen(data=data),
            status_code=200,
            media_type="text/event-stream"
        )
    else:
        return Response(
            status_code=200,
            content=codegen(data=data),
            media_type="application/json"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
