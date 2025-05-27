from collections import deque
from copy import deepcopy
import json
import logging
from typing import TypeVar

from jsonschema import validate
from pydantic import BaseModel

from btdcore.rest_client_base import RestClientBase
from btdcore.utils import scrub_title_key

from expert_llm.models import LlmChatClient, ChatBlock, LlmEmbeddingClient


DEFAULT_MAX_TOKENS = 100_000
DEFAULT_TEMPERATURE = 0.1


T = TypeVar("T", bound=BaseModel)


def format_schema_all_properties_required(schema: dict) -> dict:
    schema = deepcopy(schema)
    q = deque([schema])
    while q:
        cur = q.popleft()
        cur["additionalProperties"] = False
        cur["required"] = list(cur["properties"].keys())
        for prop_conf in cur["properties"].values():
            if prop_conf["type"] == "object":
                q.append(prop_conf)
                pass
            if prop_conf["type"] == "array":
                if prop_conf["items"]["type"] == "object":
                    q.append(prop_conf["items"])
                    pass
                pass
            pass
        pass
    return schema


class OpenAiShapedClient(LlmChatClient, LlmEmbeddingClient):
    def __init__(
        self,
        base: str,
        model: str,
        headers: dict,
        schema_requires_all_properties=False,
        rate_limit_window_seconds=1,
        rate_limit_requests=90,
        supports_true_json_mode: bool = True,
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
        self.schema_requires_all_properties = schema_requires_all_properties
        self.supports_true_json_mode = supports_true_json_mode
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
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [block.dump_for_prompt() for block in chat_blocks],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
            pass
        for allowed_key in ["service_tier"]:
            if allowed_key in kwargs:
                payload[allowed_key] = kwargs[allowed_key]
            pass
        return payload

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
        do_validate: bool = True,
        **kwargs,
    ) -> dict:
        if not self.supports_true_json_mode:
            # have to shim the schema def in
            system_messages = [block for block in chat_blocks if block.role == "system"]
            if not system_messages:
                chat_blocks = [
                    ChatBlock(
                        content="",
                        role="system",
                    ),
                    *chat_blocks,
                ]
                system_messages = chat_blocks[:1]
                pass
            system_message = system_messages[-1]
            system_message.content = "\n".join(
                [
                    system_message.content,
                    "Your response must conform to the following JSON schema:",
                    json.dumps(output_schema),
                ]
            )
            pass
        payload = self._get_base_payload(chat_blocks, **kwargs)
        if self.supports_true_json_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema_name or "Output",
                    "schema": (
                        output_schema
                        if not self.schema_requires_all_properties
                        else format_schema_all_properties_required(output_schema)
                    ),
                    "strict": kwargs.get("strict", True),
                },
            }
            pass
        else:
            payload["response_format"] = {"type": "json_object"}
            pass

        r = self.client._req("POST", "/chat/completions", json=payload)
        raw = r.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw)
            if do_validate:
                validate(parsed, output_schema)
                pass
            return parsed
        except json.JSONDecodeError as e:
            logging.error("failed to parse to JSON error: %s", e)
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = self.client._req(
            "POST",
            "/embeddings",
            json={
                "model": self.model,
                "input": texts,
            },
        )
        return [r["embedding"] for r in res.json()["data"]]

    def elect_tool_calls(
            self,
            chat_blocks: list[ChatBlock],
            tools: list[dict],
            **kwargs,
    ) -> list[dict]:
        payload = self._get_base_payload(chat_blocks, tools=tools, **kwargs)
        r = self.client._req("POST", "/chat/completions", json=payload)
        message = r.json()["choices"][0]["message"]
        return message["tool_calls"]
