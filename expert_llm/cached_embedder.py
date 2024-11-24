import logging
from multiprocessing import Lock
from pathlib import Path
import shelve

from btdcore.utils import md5_b64_str

from expert_llm.models import LlmEmbeddingClient


class CachedEmbedder:
    CACHE_FILE_NAME = ".embeddings_cache.dat"

    def __init__(
            self,
            *,
            client: LlmEmbeddingClient,
            cache_dir: Path,
    ):
        self.client = client
        self.cache_file = cache_dir / self.CACHE_FILE_NAME
        self.lock = Lock()
        return

    def _lookup_text(self, shelf: shelve.Shelf, text: str) -> list[float] | None:
        text_hash = md5_b64_str(text)
        if text_hash in shelf:
            return shelf[text_hash]
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        cached: list[
            list[float] | None
        ] = []
        with self.lock:
            with shelve.open(self.cache_file) as shelf:
                cached = [
                    self._lookup_text(shelf, text)
                    for text in texts
                ]
                pass
            pass
        to_compute = {
            text: i
            for i, text in enumerate(texts)
            if cached[i] is None
        }
        if not to_compute:
            return cached
        new_embeddings = self.client.embed(list(to_compute.keys()))
        with self.lock:
            with shelve.open(self.cache_file) as shelf:
                for text, new_embed in zip(to_compute.keys(), new_embeddings):
                    hashed_text = md5_b64_str(text)
                    shelf[hashed_text] = new_embed
                    pass
                pass
            pass
        for text, new_embed in zip(to_compute.keys(), new_embeddings):
            cached[to_compute[text]] = new_embed
            pass
        return cached

    pass
