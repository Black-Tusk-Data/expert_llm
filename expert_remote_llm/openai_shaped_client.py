from typing import TypeVar

from pydantic import BaseModel
from btdcore.rest_client_base import RestClientBase
from btdcore.utils import scrub_title_key

from .models import ChatBlock


DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.1


T = TypeVar("T", bound=BaseModel)


class OpenAiShapedClient(RestClientBase):
    def __init__(
        self,
        base: str,
        model: str,
        headers: dict,
        rate_limit_window_seconds=1,
        rate_limit_requests=90,
    ) -> None:
        super().__init__(
            base=base,
            headers=headers,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_requests=rate_limit_requests,
        )
        self.model = model
        return

    def _get_base_payload(
        self,
        chat_blocks: list[ChatBlock],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict:
        return {
            "model": self.model,
            "messages": [block.model_dump() for block in chat_blocks],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def chat_completion(
        self,
        chat_blocks: list[ChatBlock],
        **kwargs,
    ) -> ChatBlock:
        payload = self._get_base_payload(chat_blocks, **kwargs)
        r = self._req("POST", "/chat/completions", json=payload)
        response = r.json()["choices"][0]["message"]
        return ChatBlock.model_validate(response)

    def structured_completion(
        self,
        chat_blocks: list[ChatBlock],
        output_model: type[T],
        **kwargs,
    ) -> T:
        schema = scrub_title_key(output_model.model_json_schema())
        payload = self._get_base_payload(chat_blocks, **kwargs)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "schema": schema,
            },
        }
        r = self._req("POST", "/chat/completions", json=payload)
        raw = r.json()["choices"][0]["message"]["content"]
        return output_model.model_validate_json(raw)

    def compute_embedding(self, text: str) -> list[float]:
        r = self._req(
            "POST",
            "/embeddings",
            json={
                "model": self.model,
                "input": text,
            },
        )
        return r.json()["data"][0]["embedding"]
