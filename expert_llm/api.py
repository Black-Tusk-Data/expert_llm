import logging
import time
import traceback

from expert_llm.models import ChatBlock, LlmChatClient, LlmResponse


class LlmApi:
    SLEEP_ON_ERR = 3

    def __init__(
        self,
        client: LlmChatClient,
        *,
        req_log_level: str = "info",
        req_kwargs: dict | None = None,
    ):
        self.llm_client = client
        self.req_log_level = req_log_level
        self.req_kwargs = req_kwargs
        return

    def elect_tool_calls(
            self,
            *,
            req_name: str,
            system: str,
            user: str,
            tools: list[dict],
            max_tokens: int = 16000,
            max_attempts: int = 3,
    ) -> list[dict]:
        def do_req() -> list[dict]:
            getattr(logging, self.req_log_level)("%s", req_name)
            return self.llm_client.elect_tool_calls(
              chat_blocks = [
                  ChatBlock(
                      role="system",
                      content=system,
                  ),
                  ChatBlock(
                      role="user",
                      content=user,
                  ),
              ],
              max_tokens=max_tokens,
              tools=tools,
            )

        return self._attempt_multiple_times(do_req, max_attempts=max_attempts)


    def completion(
        self,
        *,
        req_name: str,
        system: str,
        user: str,
        output_schema: dict | None = None,
        max_tokens: int | None = None,
        max_attempts: int = 3,
        **kwargs,
    ) -> LlmResponse:
        kwargs = {
            **(self.req_kwargs or {}),
            **kwargs,
        }
        chat_blocks = [
            ChatBlock(
                role="system",
                content=system,
            ),
            ChatBlock(
                role="user",
                content=user,
            ),
        ]

        def do_req() -> LlmResponse:
            getattr(logging, self.req_log_level)("%s", req_name)
            if output_schema:
                output = self.llm_client.structured_completion_raw(
                    chat_blocks=chat_blocks,
                    output_schema=output_schema,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return LlmResponse(
                    structured_output=output,
                )
            completion = self.llm_client.chat_completion(
                chat_blocks, max_tokens=max_tokens
            )
            return LlmResponse(message=completion.content)

        return self._attempt_multiple_times(do_req, max_attempts=max_attempts)

    def _attempt_multiple_times(
            self,
            req_fn,
            *,
            max_attempts: int,
    ):
        last_err: Exception
        for attempt in range(max_attempts):
            try:
                return req_fn()
            except Exception as e:
                last_err = e
                logging.error(
                    "LLM request failed: %s, trace: %s", e, traceback.format_exc()
                )
                if attempt + 1 == max_attempts:
                    raise e
                time.sleep(self.SLEEP_ON_ERR)
                pass
            pass

        # never hit, just for typing
        raise last_err
        return

    pass
