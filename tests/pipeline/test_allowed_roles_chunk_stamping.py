"""S2 of the document-based RBAC layer (see ``ACL_PLAN.md``).

Two properties under test:

1. **build_chunks_dict_from_chunking_result** stamps each chunk with a
   *copied* ``roles`` list when ``allowed_roles`` is supplied, and leaves the
   ``roles`` key absent (open) otherwise.

2. **end-to-end stamping**: a document inserted with ``allowed_roles`` lands
   roles on its persisted text-chunk records (verified on the default JSON-KV
   backend via ``text_chunks.get_by_id``), and an open document does not.

The PG-specific schema/SQL (roles column, GIN index, && pre-filter) is
exercised by the DB-backed tests (``requires_db``); here we cover the
backend-agnostic stamping path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface
from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result


class _SimpleTokenizerImpl(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt, **kwargs):
    return "ENTITY|x|concept|desc"


_ROLE_FIELD_SUFFIXES = (
    ("_llm_model_func", "func"),
    ("_llm_model_kwargs", "kwargs"),
    ("_llm_model_max_async", "max_async"),
    ("_llm_timeout", "timeout"),
)


def _new_rag(tmp_path: Path, **kwargs) -> LightRAG:
    role_configs: dict[str, RoleLLMConfig] = {}
    for spec in ROLES:
        bucket = {}
        for suffix, target in _ROLE_FIELD_SUFFIXES:
            key = f"{spec.name}{suffix}"
            if key in kwargs:
                bucket[target] = kwargs.pop(key)
        if bucket:
            role_configs[spec.name] = RoleLLMConfig(**bucket)
    if role_configs:
        kwargs["role_llm_configs"] = role_configs

    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"acl-chunks-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 1. Pure builder
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_build_chunks_stamps_roles_copy():
    chunking_result = [
        {"content": "alpha", "chunk_order_index": 0, "tokens": 1},
        {"content": "beta", "chunk_order_index": 1, "tokens": 1},
    ]
    roles = ["public", "confidential"]
    chunks = build_chunks_dict_from_chunking_result(
        chunking_result, doc_id="doc-x", file_path="x.txt", allowed_roles=roles
    )
    assert len(chunks) == 2
    for rec in chunks.values():
        assert rec["roles"] == ["public", "confidential"]
        # Must be a copy — mutating the source list must not bleed in.
        assert rec["roles"] is not roles
    roles.append("leak")
    assert all("leak" not in rec["roles"] for rec in chunks.values())


@pytest.mark.offline
def test_build_chunks_open_when_no_roles():
    chunking_result = [{"content": "alpha", "chunk_order_index": 0, "tokens": 1}]
    for kw in ({}, {"allowed_roles": None}, {"allowed_roles": []}):
        chunks = build_chunks_dict_from_chunking_result(
            chunking_result, doc_id="doc-y", file_path="y.txt", **kw
        )
        rec = next(iter(chunks.values()))
        assert "roles" not in rec


# --------------------------------------------------------------------------- #
# 2. End-to-end stamping onto persisted text chunks (JSON-KV backend)
# --------------------------------------------------------------------------- #


def _chunk_records_for_doc(rag: LightRAG, doc_id: str) -> list[dict]:
    """Return persisted text-chunk records belonging to ``doc_id``."""
    store = rag.text_chunks._data  # JsonKVStorage in-memory dict
    return [
        {**v, "_id": cid}
        for cid, v in store.items()
        if v.get("full_doc_id") == doc_id
    ]


@pytest.mark.offline
def test_roles_stamped_on_persisted_chunks(tmp_path):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.ainsert(
                "Confidential body about ACME and project X.",
                ids=["doc-sec"],
                file_paths=["sec.txt"],
                allowed_roles=["confidential"],
            )
            await rag.ainsert(
                "Open body about ACME the manufacturer.",
                ids=["doc-open"],
                file_paths=["open.txt"],
            )
            return (
                _chunk_records_for_doc(rag, "doc-sec"),
                _chunk_records_for_doc(rag, "doc-open"),
            )
        finally:
            await rag.finalize_storages()

    sec_chunks, open_chunks = asyncio.run(_run())
    assert sec_chunks, "secured doc must have produced chunks"
    assert open_chunks, "open doc must have produced chunks"
    assert all(c.get("roles") == ["confidential"] for c in sec_chunks)
    # Open doc: no roles stamped.
    assert all(not c.get("roles") for c in open_chunks)
