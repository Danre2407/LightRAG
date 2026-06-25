"""S1 of the document-based RBAC layer (see ``ACL_PLAN.md``).

Two properties under test:

1. **normalize_allowed_roles**: the pure canonicalizer accepts a single
   string or an iterable of strings, strips/dedupes/drops-empties, and
   returns ``None`` (the "open" sentinel) when nothing usable remains.

2. **enqueue persistence**: ``allowed_roles`` passed to
   ``apipeline_enqueue_documents`` (single list broadcast, or a
   ``list[list[str]]`` aligned per-document) lands on
   ``doc_status.metadata['allowed_roles']`` — and is absent for documents
   enqueued without roles, preserving the pre-RBAC "open" default.

No enforcement is exercised here; S1 only threads + persists the roles.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface
from lightrag.utils_pipeline import normalize_allowed_roles


class _SimpleTokenizerImpl(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt, **kwargs):
    return '{"name":"x","summary":"s","detail_description":"d"}'


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
        workspace=f"acl-roles-{tmp_path.name}",
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
# 1. Pure normalizer
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_normalize_allowed_roles_none_and_empty():
    assert normalize_allowed_roles(None) is None
    assert normalize_allowed_roles([]) is None
    assert normalize_allowed_roles(["", "  "]) is None
    assert normalize_allowed_roles("") is None


@pytest.mark.offline
def test_normalize_allowed_roles_string_and_iterables():
    assert normalize_allowed_roles("public") == ["public"]
    # strip + dedupe preserving first-seen order
    assert normalize_allowed_roles(["public", "public", "  conf "]) == [
        "public",
        "conf",
    ]
    assert normalize_allowed_roles(("a", "b", "a")) == ["a", "b"]


@pytest.mark.offline
def test_normalize_allowed_roles_rejects_non_strings():
    with pytest.raises(ValueError):
        normalize_allowed_roles([1, 2])
    with pytest.raises(ValueError):
        normalize_allowed_roles(5)


# --------------------------------------------------------------------------- #
# 2. Enqueue persistence into doc_status.metadata
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_allowed_roles_broadcast_persisted(tmp_path):
    """A single list[str] is broadcast to every document's metadata."""

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                ["doc one body", "doc two body"],
                ids=["doc-acl-a", "doc-acl-b"],
                file_paths=["a.txt", "b.txt"],
                track_id="track-acl-broadcast",
                allowed_roles=["public", "public", "confidential"],
            )
            row_a = await rag.doc_status.get_by_id("doc-acl-a")
            row_b = await rag.doc_status.get_by_id("doc-acl-b")
        finally:
            await rag.finalize_storages()
        return row_a, row_b

    row_a, row_b = asyncio.run(_run())
    assert row_a is not None and row_b is not None
    # Deduped + order preserved by the normalizer.
    assert row_a["metadata"]["allowed_roles"] == ["public", "confidential"]
    assert row_b["metadata"]["allowed_roles"] == ["public", "confidential"]


@pytest.mark.offline
def test_allowed_roles_per_document_list(tmp_path):
    """A list[list[str]] assigns independent roles per document."""

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                ["pub body", "sec body"],
                ids=["doc-acl-pub", "doc-acl-sec"],
                file_paths=["pub.txt", "sec.txt"],
                track_id="track-acl-perdoc",
                allowed_roles=[["public"], ["confidential"]],
            )
            row_pub = await rag.doc_status.get_by_id("doc-acl-pub")
            row_sec = await rag.doc_status.get_by_id("doc-acl-sec")
        finally:
            await rag.finalize_storages()
        return row_pub, row_sec

    row_pub, row_sec = asyncio.run(_run())
    assert row_pub["metadata"]["allowed_roles"] == ["public"]
    assert row_sec["metadata"]["allowed_roles"] == ["confidential"]


@pytest.mark.offline
def test_allowed_roles_absent_is_open(tmp_path):
    """Without allowed_roles, metadata carries no roles key (open default)."""

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "open doc body",
                ids=["doc-acl-open"],
                file_paths=["open.txt"],
                track_id="track-acl-open",
            )
            return await rag.doc_status.get_by_id("doc-acl-open")
        finally:
            await rag.finalize_storages()

    row = asyncio.run(_run())
    assert row is not None
    assert "allowed_roles" not in (row.get("metadata") or {})


@pytest.mark.offline
def test_allowed_roles_length_mismatch_raises(tmp_path):
    """A per-document roles list misaligned with input is rejected."""

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            with pytest.raises(ValueError):
                await rag.apipeline_enqueue_documents(
                    ["only one body"],
                    ids=["doc-acl-mismatch"],
                    file_paths=["m.txt"],
                    track_id="track-acl-mismatch",
                    allowed_roles=[["public"], ["confidential"]],
                )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
