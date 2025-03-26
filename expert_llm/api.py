from expert_llm.models import ChatBlock, LlmChatClient, LlmResponse


class LlmApi:
    def __init__(self, client: LlmChatClient):
        self.llm_client = client
        return

    def completion(
        self,
        *,
        req_name: str,
        system: str,
        user: str,
        output_schema: dict | None = None,
        max_tokens: int = 16000,
    ) -> LlmResponse:
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
        result: LlmResponse
        if output_schema:
            output = self.llm_client.structured_completion_raw(
                chat_blocks=chat_blocks,
                output_schema=output_schema,
                max_tokens=max_tokens,
            )
            result = LlmResponse(
                structured_output=output,
            )
            pass
        else:
            completion = self.llm_client.chat_completion(
                chat_blocks, max_tokens=max_tokens
            )
            result = LlmResponse(message=completion.content)
            pass
        return result

    pass
