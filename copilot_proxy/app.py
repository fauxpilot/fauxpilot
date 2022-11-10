import os
from typing import Dict

import ujson
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException, status, Response, Depends
from firebase_admin import auth, credentials, initialize_app
from starlette.responses import FileResponse

from models import OpenAIinput
from utils.codegen import CodeGenProxy

SHOULD_AUTHENTICATE = os.getenv('SHOULD_AUTHENTICATE', 'false').lower() == 'true'

codegen = CodeGenProxy(
    # host=os.environ.get("TRITON_HOST", "10.128.15.213"), 40GB
    host=os.environ.get("TRITON_HOST", "10.128.15.246"), # 80GB
    port=os.environ.get("TRITON_PORT", 8001),
    verbose=os.environ.get("TRITON_VERBOSITY", False)
)

app = FastAPI(
    title="FauxPilot",
    description="This is an attempt to build a locally hosted version of GitHub Copilot. It uses the SalesForce CodeGen"
                "models inside of NVIDIA's Triton Inference Server with the FasterTransformer backend.",
    docs_url="/",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

try:
    admin_credential = credentials.Certificate('../serviceAccountKey.json')
    initialize_app(admin_credential)
except (ValueError, FileNotFoundError):
    pass


def get_user_token(res: Response, credential: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
    if not SHOULD_AUTHENTICATE:
        return None
    if credential is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is needed",
            headers={'WWW-Authenticate': 'Bearer realm="auth_required"'},
        )
    try:
        user = auth.verify_id_token(credential.credentials)
        if not auth.get_user_by_email(user['email']).custom_claims['allowed']:
            raise HTTPException
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication from Firebase. {err}",
            headers={'WWW-Authenticate': 'Bearer error="invalid_token"'},
        )
    res.headers['WWW-Authenticate'] = 'Bearer realm="auth_required"'
    return user


class CustomTokenRequest(BaseModel):
    id_token: str


class PlaygroundRequest(BaseModel):
    prompt: str
    max_tokens: int = 32


class PlaygroundResponse(BaseModel):
    results: str
    metadata: dict

@app.post("/v1/auth/get_custom_token", include_in_schema=False)
async def get_custom_token(custom_token_request: CustomTokenRequest):
    try:
        user = auth.verify_id_token(custom_token_request.id_token)
        if not auth.get_user_by_email(user['email']).custom_claims['allowed']:
            raise HTTPException
        return {
            "id_token": auth.create_custom_token(user["uid"])
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication from Firebase. {err}",
            headers={'WWW-Authenticate': 'Bearer error="invalid_id_token"'},
        )


@app.get("/v1/playground/get_config", status_code=200, include_in_schema=False)
async def playground() -> Dict:
    default_config = OpenAIinput().dict()
    del default_config["prompt"]
    return default_config


@app.post("/v1/playground", status_code=200, include_in_schema=False)
async def playground(data: OpenAIinput):
    results = await codegen(data.dict())
    parsed_results = ujson.loads(results)
    return Response(
        status_code=200,
        content=parsed_results["choices"][0]["text"],
        media_type="application/text"
    )


@app.get("/playground")
async def read_index():
    return FileResponse('index.html')


@app.post("/v1/engines/codegen/completions", status_code=200)
@app.post("/v1/completions", status_code=200)
async def completions(data: OpenAIinput, user=Depends(get_user_token)):
    results = await codegen(data.dict())
    if data.stream is not None:
        return EventSourceResponse(
            content=results,
            status_code=200,
            media_type="text/event-stream"
        )
    else:
        return Response(
            status_code=200,
            content=results,
            media_type="application/json"
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5432)
