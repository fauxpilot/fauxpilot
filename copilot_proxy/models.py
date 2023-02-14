from typing import Optional, Union

from pydantic import BaseModel, constr

ModelType = constr(regex="^(fastertransformer|py-model)$")


class OpenAIinput(BaseModel):
    model: ModelType = "fastertransformer"
    prompt: Optional[str]
    suffix: Optional[str]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool]
    logprobs: Optional[int] = None
    echo: Optional[bool]
    stop: Optional[Union[str, list]]
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 1
    best_of: Optional[int] = 1
    logit_bias: Optional[dict]
    user: Optional[str]
