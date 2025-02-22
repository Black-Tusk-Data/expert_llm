import os
from typing import Literal

from .openai_shaped_client import OpenAiShapedClient


OpenAiModel = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-4o-2024-08-06",
    "text-embedding-3-small",
]
OctoModel = Literal[
    "mixtral-8x7b-instruct-fp16",
    "nous-hermes-2-mixtral-8x7b-dpo",
    "meta-llama-3.1-70b-instruct",
    "meta-llama-3.1-8b-instruct",
]
TogetherAiModel = Literal[
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
]
GroqModel = Literal[
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llava-v1.5-7b-4096-preview",
]


class OpenAIApiClient(OpenAiShapedClient):
    def __init__(self, model: OpenAiModel, **kwargs) -> None:
        api_key = os.environ["OPENAI_API_KEY"]
        super().__init__(
            model=str(model),
            base="https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "assistants=v1",
            },
            # pretty rough, this is not enforced as specified here
            rate_limit_window_seconds=1,
            rate_limit_requests=5,
            **kwargs,
        )
        return

    pass


class TogetherAiClient(OpenAiShapedClient):
    def __init__(self, model: TogetherAiModel, **kwargs) -> None:
        API_KEY = os.environ["TOGETHER_API_KEY"]
        super().__init__(
            model=str(model),
            base="https://api.together.xyz/v1",
            headers={"Authorization": f"Bearer {API_KEY}"},
            rate_limit_window_seconds=1,
            rate_limit_requests=10,
            **kwargs,
        )
        return

    pass


class GroqClient(OpenAiShapedClient):
    def __init__(self, model: GroqModel, **kwargs) -> None:
        # enforcing a basic rate limit targeting 30 reqs/min

        API_KEY = os.environ["GROQ_API_KEY"]
        super().__init__(
            base="https://api.groq.com/openai/v1",
            model=model,
            headers={"Authorization": f"Bearer {API_KEY}"},
            rate_limit_window_seconds=2,
            rate_limit_requests=1,
            **kwargs,
        )
        return

    # def get_rate_limit(self, model: GroqModel) -> tuple[int, int]:
    #     if model == "llama-3.1-8b-instant":
    #         return
    #     if model == "llama-3.1-70b-versatile":
    #         return
    #     if model == "llava-v1.5-7b-4096-preview":
    #         return
    #     return

    pass
