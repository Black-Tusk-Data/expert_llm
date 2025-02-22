import json
import logging
from typing import TypeVar

from pydantic import BaseModel
from btdcore.rest_client_base import RestClientBase
from btdcore.utils import scrub_title_key

from expert_llm.models import LlmChatClient, ChatBlock


DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.1


T = TypeVar("T", bound=BaseModel)


class OpenAiShapedClient(LlmChatClient):
    def __init__(
        self,
        base: str,
        model: str,
        headers: dict,
        rate_limit_window_seconds=1,
        rate_limit_requests=90,
        **kwargs,
    ) -> None:
        self.base = base
        self.headers = headers
        self.client = RestClientBase(
            base=base,
            headers=headers,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_requests=rate_limit_requests,
            **kwargs,
        )
        self.model = model
        self.max_concurrent_requests = rate_limit_requests // rate_limit_window_seconds
        return

    def override_rate_limit(
            self,
            *,
            rate_limit_window_seconds: int,
            rate_limit_requests: int,
    ):
        self.max_concurrent_requests = rate_limit_requests // rate_limit_window_seconds
        self.client = RestClientBase(
            base=self.base,
            headers=self.headers,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_requests=rate_limit_requests,
        )
        return


    def get_max_concurrent_requests(self) -> int:
        return self.max_concurrent_requests

    def _get_base_payload(
        self,
        chat_blocks: list[ChatBlock],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict:
        return {
            "model": self.model,
            "messages": [block.dump_for_prompt() for block in chat_blocks],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def chat_completion(
        self,
        chat_blocks: list[ChatBlock],
        **kwargs,
    ) -> ChatBlock:
        payload = self._get_base_payload(chat_blocks, **kwargs)
        r = self.client._req("POST", "/chat/completions", json=payload)
        response = r.json()["choices"][0]["message"]
        return ChatBlock.model_validate(response)

    def structured_completion_raw(
        self,
        *,
        chat_blocks: list[ChatBlock],
        output_schema: dict,
        output_schema_name: str | None = None,
        **kwargs,
    ) -> dict:
        payload = self._get_base_payload(chat_blocks, **kwargs)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema_name or "Output",
                "schema": output_schema,
            },
        }
        r = self.client._req("POST", "/chat/completions", json=payload)
        raw = r.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error("failed to parse to JSON: %s, error: %s", raw, e)
            raise e
        pass

    def structured_completion(
        self,
        chat_blocks: list[ChatBlock],
        output_model: type[T],
        **kwargs,
    ) -> T:
        schema = scrub_title_key(output_model.model_json_schema())
        raw = self.structured_completion_raw(
            chat_blocks=chat_blocks,
            output_schema=schema,
            output_schema_name=output_model.__name__,
        )
        return output_model.model_validate(raw)

    def compute_embedding(self, text: str) -> list[float]:
        r = self.client._req(
            "POST",
            "/embeddings",
            json={
                "model": self.model,
                "input": text,
            },
        )
        return r.json()["data"][0]["embedding"]
