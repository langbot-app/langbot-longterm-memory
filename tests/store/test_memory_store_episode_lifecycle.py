from __future__ import annotations

import pytest

from store.memory_store import MemoryStore


class FakeVectorPlugin:
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.upserts: list[tuple[str, dict]] = []

    async def invoke_embedding(self, _embedding_model_uuid: str, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] for index, _ in enumerate(texts)]

    async def vector_upsert(self, collection_id, vectors, ids, metadata=None, documents=None):
        for index, item_id in enumerate(ids):
            meta = dict(metadata[index])
            self.records[item_id] = {
                "id": item_id,
                "metadata": meta,
                "document": documents[index],
                "distance": 0.0,
                "score": 1.0,
            }
            self.upserts.append((item_id, meta))

    async def vector_search(self, collection_id, query_vector, top_k=5, filters=None, **_kwargs):
        return self._filtered(filters)[:top_k]

    async def vector_list(self, collection_id, filters=None, limit=20, offset=0):
        items = self._filtered(filters)
        return {"items": items[offset: offset + limit], "total": len(items)}

    def _filtered(self, filters=None):
        items = list(self.records.values())
        if not filters:
            return items

        conditions = filters.get("$and") if isinstance(filters, dict) else None
        if conditions is None:
            conditions = [filters]

        for condition in conditions:
            for key, value in condition.items():
                if isinstance(value, dict):
                    continue
                items = [
                    item
                    for item in items
                    if item["metadata"].get(key) == value
                ]
        return items


def _record(
    item_id: str,
    *,
    user_key: str = "user-1",
    status: str | None = "active",
    content: str | None = None,
) -> dict:
    metadata = {
        "content": content or f"memory {item_id}",
        "tags": "",
        "importance": "2",
        "timestamp": "2026-01-01T00:00:00Z",
        "user_key": user_key,
        "source": "agent",
    }
    if status is not None:
        metadata["status"] = status
    return {
        "id": item_id,
        "metadata": metadata,
        "document": metadata["content"],
        "distance": 0.0,
        "score": 1.0,
    }


@pytest.mark.asyncio
async def test_add_episode_defaults_to_active_status():
    plugin = FakeVectorPlugin()
    store = MemoryStore(plugin)

    episode = await store.add_episode(
        collection_id="kb-1",
        embedding_model_uuid="emb-1",
        user_key="user-1",
        content="User likes concise answers",
    )

    assert episode["status"] == "active"
    stored = next(iter(plugin.records.values()))
    assert stored["metadata"]["status"] == "active"


@pytest.mark.asyncio
async def test_search_and_list_default_to_active_and_legacy_records():
    plugin = FakeVectorPlugin()
    plugin.records = {
        "active-1": _record("active-1", status="active"),
        "legacy-1": _record("legacy-1", status=None),
        "superseded-1": _record("superseded-1", status="superseded"),
        "archived-1": _record("archived-1", status="archived"),
    }
    store = MemoryStore(plugin)

    search_results = await store.search_episodes(
        collection_id="kb-1",
        embedding_model_uuid="emb-1",
        query="memory",
        user_key="user-1",
        top_k=10,
    )
    listed, total = await store.list_episodes(
        collection_id="kb-1",
        user_key="user-1",
        limit=10,
        offset=0,
    )

    assert {item["id"] for item in search_results} == {"active-1", "legacy-1"}
    assert {item["id"] for item in listed} == {"active-1", "legacy-1"}
    assert total == 2


@pytest.mark.asyncio
async def test_include_superseded_can_inspect_hidden_records():
    plugin = FakeVectorPlugin()
    plugin.records = {
        "active-1": _record("active-1", status="active"),
        "superseded-1": _record("superseded-1", status="superseded"),
    }
    store = MemoryStore(plugin)

    results = await store.search_episodes(
        collection_id="kb-1",
        embedding_model_uuid="emb-1",
        query="memory",
        user_key="user-1",
        top_k=10,
        include_statuses=["active", "superseded"],
    )

    assert {item["id"] for item in results} == {"active-1", "superseded-1"}
    assert next(item for item in results if item["id"] == "superseded-1")["status"] == "superseded"


@pytest.mark.asyncio
async def test_auto_supersede_marks_old_episode_status():
    plugin = FakeVectorPlugin()
    plugin.records = {
        "old-1": _record("old-1", status="active", content="User prefers tea"),
    }
    store = MemoryStore(plugin)

    await store.add_episode(
        collection_id="kb-1",
        embedding_model_uuid="emb-1",
        user_key="user-1",
        content="Correction: user prefers coffee",
        tags=["correction"],
    )

    assert plugin.records["old-1"]["metadata"]["status"] == "superseded"
    assert plugin.records["old-1"]["metadata"]["superseded_by"]
