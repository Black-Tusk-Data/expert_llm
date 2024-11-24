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
        to_compute = [
            (text, i)
            for i, text in enumerate(texts)
            if cached[i] is None
        ]
        if not to_compute:
            return cached
        new_embeddings = self.client.embed([
            text
            for text, _ in to_compute
        ])
        with self.lock:
            with shelve.open(self.cache_file) as shelf:
                for (text, _), new_embed in zip(to_compute, new_embeddings):
                    hashed_text = md5_b64_str(text)
                    try:
                        shelf[hashed_text] = new_embed
                    except Exception as e:
                        logging.error("failed to add embedding to cache: %s", e)
                        pass
                    pass
                pass
            pass

        results = [*cached]
        for (_, i), new_embed in zip(to_compute, new_embeddings):
            results[i] = new_embed
            pass

        assert all(results)
        return results

    pass
