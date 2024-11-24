import os
from typing import Literal

from btdcore.rest_client_base import RestClientBase
from expert_llm.models import LlmEmbeddingClient


JinaModel = Literal[
    # "jina-clip-v1",
    "jina-embeddings-v2-base-en",
    "jina-embeddings-v3",
    # "jina-embeddings-v2-base-code",
]


class JinaAiClient(LlmEmbeddingClient):
    def __init__(
        self,
        model: JinaModel,
    ):
        self.model = model
        JINA_AI_API_KEY = os.environ["JINA_AI_API_KEY"]
        self.client = RestClientBase(
            base="https://api.jina.ai/v1",
            headers=dict(Authorization=f"Bearer {JINA_AI_API_KEY}"),
            rate_limit_window_seconds=1,
            rate_limit_requests=2,
        )
        return

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = self.client._req(
            "POST",
            "/embeddings",
            json={
                "input": texts,
                "model": self.model,
            },
        )
        if not res.ok:
            res.raise_for_status()
        data = res.json()
        embeds = [r["embedding"] for r in data["data"]]
        return embeds

    def get_embedding_vector_length(self) -> int:
        if self.model == "jina-embeddings-v2-base-en":
            return 768
        return super().get_embedding_vector_length()

    pass
