import os

import uvicorn
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException, status, Response, Depends
from firebase_admin import auth, credentials, initialize_app

from models import OpenAIinput
from utils.codegen import CodeGenProxy

codegen = CodeGenProxy(
    host=os.environ.get("TRITON_HOST", "fauxpilot-triton.codium-inc.com"),
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

try:
    admin_credential = credentials.Certificate('../serviceAccountKey.json')
    initialize_app(admin_credential)
except ValueError:
    pass


def get_user_token(res: Response, credential: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
    if credential is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is needed",
            headers={'WWW-Authenticate': 'Bearer realm="auth_required"'},
        )
    try:
        decoded_token = auth.verify_id_token(credential.credentials)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication from Firebase. {err}",
            headers={'WWW-Authenticate': 'Bearer error="invalid_token"'},
        )
    res.headers['WWW-Authenticate'] = 'Bearer realm="auth_required"'
    return decoded_token


@app.get("/v1/auth/login")
async def hello_user(id_token: str):
    try:
        return auth.create_custom_token(id_token)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication from Firebase. {err}",
            headers={'WWW-Authenticate': 'Bearer error="invalid_id_token"'},
        )


@app.post("/v1/engines/codegen/completions", status_code=200)
@app.post("/v1/completions", status_code=200)
async def completions(data: OpenAIinput, user=Depends(get_user_token)):
    data = data.dict()
    print(user)
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
    uvicorn.run("app:app", host="0.0.0.0", port=5432)
