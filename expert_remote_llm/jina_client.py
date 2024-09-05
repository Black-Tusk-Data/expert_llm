import os
from typing import Literal

from btdcore.rest_client_base import RestClientBase


JinaModel = Literal[
    "jina-clip-v1",
    "jina-embeddings-v2-base-en",
    "jina-embeddings-v2-base-code",
]


class JinaClient(RestClientBase):
    def __init__(
        self,
        model: JinaModel,
    ):
        self.model = model
        JINA_AI_API_KEY = os.environ["JINA_AI_API_KEY"]
        super().__init__(
            base="https://api.jina.ai/v1",
            headers=dict(Authorization=f"Bearer {JINA_AI_API_KEY}"),
            rate_limit_window_seconds=1,
            rate_limit_requests=2,
        )
        return

    def embed(self, texts: list[str]) -> list[float]:
        res = self._req(
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

    pass
