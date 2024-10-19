import abc
from typing import Literal, TypeVar

from pydantic import BaseModel


ChatRole = Literal["system", "user", "assistant"]

T = TypeVar("T", bound=BaseModel)


class ChatBlock(BaseModel):
    role: ChatRole
    content: str
    image_b64: str | None = None

    def dump_for_prompt(self) -> dict:
        if not self.image_b64:
            return {
                "role": self.role,
                "content": self.content,
            }
        # o/w we have to change the format a bit
        return {
            "role": self.role,
            "content": [
                {"type": "text", "text": self.content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.image_b64}",
                    },
                },
            ],
        }

    pass


class LlmChatClient(abc.ABC):
    @abc.abstractmethod
    def get_max_concurrent_requests(self) -> int:
        pass

    @abc.abstractmethod
    def chat_completion(
        self,
        chat_blocks: list[ChatBlock],
        **kwargs,
    ) -> ChatBlock:
        pass

    @abc.abstractmethod
    def structured_completion(
        self,
        chat_blocks: list[ChatBlock],
        output_model: type[T],
        **kwargs,
    ) -> T:
        pass

    pass


class LlmEmbeddingClient(abc.ABC):
    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    def get_embedding_vector_length(self) -> int:
        raise Exception("Don't know model embedding size!")

    pass
