from typing import Optional, Union

from pydantic import BaseModel


class OpenAIinput(BaseModel):
    model: str = "fastertransformer"
    prompt: Optional[str] = "Hello world"
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    stop: Optional[Union[str, list]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 1
    best_of: Optional[int] = 1
    logit_bias: Optional[dict] = None
    user: Optional[str] = None

