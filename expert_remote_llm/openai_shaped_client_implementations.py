import os
from typing import Literal

from .openai_shaped_client import OpenAiShapedClient


OpenAiModel = Literal[
    "gpt-4o", "gpt-3.5-turbo", "gpt-4o-2024-08-06", "text-embedding-3-small"
]
OctoModel = Literal[
    "mixtral-8x7b-instruct-fp16",
    "nous-hermes-2-mixtral-8x7b-dpo",
]
TogetherAiModel = Literal[
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


class OpenAIApiClient(OpenAiShapedClient):
    def __init__(self, model: OpenAiModel) -> None:
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
        )
        return


class OctoAiApiClient(OpenAiShapedClient):
    def __init__(self, model: OctoModel) -> None:
        OCTOAI_API_KEY = os.environ.get("OCTOAI_API_KEY")
        super().__init__(
            model=str(model),
            base="https://text.octoai.run/v1",
            headers={"Authorization": f"Bearer {OCTOAI_API_KEY}"},
            # this is slower than advertised, but empirically enforced
            rate_limit_window_seconds=1,
            rate_limit_requests=1,
        )
        return


class TogetherAiClient(OpenAiShapedClient):
    def __init__(self, model: TogetherAiModel) -> None:
        API_KEY = os.environ.get("TOGETHER_API_KEY")
        super().__init__(
            model=str(model),
            base="https://api.together.xyz/v1",
            headers={"Authorization": f"Bearer {API_KEY}"},
            rate_limit_window_seconds=1,
            rate_limit_requests=90,
        )
        return
