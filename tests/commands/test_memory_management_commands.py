from __future__ import annotations

import json

import pytest

from components.commands.memory import Memory
from langbot_plugin.api.entities.builtin.command.context import ExecuteContext
from langbot_plugin.api.entities.builtin.provider.session import Session, LauncherTypes


class FakeAPI:
    async def get_bot_uuid(self) -> str:
        return "bot-1"

    async def get_query_vars(self) -> dict:
        return {"sender_id": "u-1", "sender_name": "Alice"}

    async def list_pipeline_knowledge_bases(self) -> list[dict]:
        return [{"uuid": "kb-1"}]


class FakeStore:
    EPISODE_STATUS_ACTIVE = "active"
    EPISODE_STATUS_SUPERSEDED = "superseded"
    EPISODE_STATUS_ARCHIVED = "archived"
    EPISODE_STATUS_DELETED = "deleted"
    EPISODE_STATUSES = {"active", "superseded", "archived", "deleted"}

    def __init__(self):
        self.audit_entries: list[dict] = []
        self.updated: list[tuple[str, str]] = []
        self.export_calls: list[dict] = []
        self.import_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    async def resolve_user_context(self, session, bot_uuid: str = ""):
        return (
            "bot-1:group_1",
            "bot-1:group_1",
            "kb-1",
            "session",
            {"embedding_model_uuid": "emb-1"},
        )

    async def get_episode_by_id(self, collection_id, episode_id, user_key):
        if episode_id != "ep-1":
            return None
        return {
            "id": "ep-1",
            "content": "Old memory",
            "tags": ["correction"],
            "importance": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "sender_id": "u-1",
            "sender_name": "Alice",
            "source": "agent",
            "status": "superseded",
            "superseded_by": "ep-2",
        }

    async def list_episodes(
        self,
        collection_id,
        user_key,
        limit=10,
        offset=0,
        include_statuses=None,
    ):
        if include_statuses == ["superseded"]:
            return [
                {
                    "id": "ep-1",
                    "content": "Old memory",
                    "tags": [],
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            ], 1
        return [], 0

    async def update_episode_status(
        self,
        collection_id,
        embedding_model_uuid,
        episode_id,
        user_key,
        status,
    ):
        if episode_id != "ep-1":
            return None
        self.updated.append((episode_id, status))
        return {"id": episode_id, "status": status}

    async def export_episodes_by_user(
        self,
        collection_id,
        user_key,
        include_statuses=None,
    ):
        self.export_calls.append({
            "collection_id": collection_id,
            "user_key": user_key,
            "include_statuses": include_statuses,
        })
        return [
            {
                "id": "ep-1",
                "content": "Old memory",
                "tags": ["correction"],
                "importance": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "user_key": user_key,
                "sender_id": "u-1",
                "sender_name": "Alice",
                "source": "agent",
                "status": "superseded",
                "superseded_by": "ep-2",
            }
        ]

    async def import_episodes_for_user(
        self,
        collection_id,
        embedding_model_uuid,
        user_key,
        episodes,
        bot_uuid="",
    ):
        self.import_calls.append({
            "collection_id": collection_id,
            "embedding_model_uuid": embedding_model_uuid,
            "user_key": user_key,
            "episodes": episodes,
            "bot_uuid": bot_uuid,
        })
        return [{"id": "new-1", "content": episodes[0]["content"], "user_key": user_key}]

    async def delete_episodes_by_filters(
        self,
        collection_id,
        user_key,
        sender_id="",
        tag="",
        before="",
        include_statuses=None,
    ):
        self.delete_calls.append({
            "collection_id": collection_id,
            "user_key": user_key,
            "sender_id": sender_id,
            "tag": tag,
            "before": before,
            "include_statuses": include_statuses,
        })
        return 2, ["archived-1", "archived-2"]

    async def append_audit_entry(self, **entry):
        self.audit_entries.append(entry)
        return entry

    @staticmethod
    def _preview_text(value: str, max_len: int = 120) -> str:
        return value[:max_len]


class FakePlugin:
    def __init__(self):
        self.memory_store = FakeStore()
        self.plugin_runtime_handler = object()


def _context(command: str, params: list[str]) -> ExecuteContext:
    return ExecuteContext(
        query_id=1,
        session=Session(launcher_type=LauncherTypes.GROUP, launcher_id="1"),
        command_text=f"memory {command}",
        full_command_text=f"!memory {command}",
        command="memory",
        crt_command=command,
        params=[command, *params],
        crt_params=params,
        privilege=0,
    )


async def _run(command: Memory, subcommand_name: str, params: list[str]) -> str:
    subcommand = command.registered_subcommands[subcommand_name].subcommand
    results = []
    async for item in subcommand(command, _context(subcommand_name, params)):
        results.append(item.text or "")
    return "\n".join(results)


@pytest.mark.asyncio
async def test_memory_show_displays_episode_details(monkeypatch):
    monkeypatch.setattr("components.commands.memory.QueryBasedAPIProxy", lambda **_: FakeAPI())
    command = Memory()
    command.plugin = FakePlugin()

    output = await _run(command, "show", ["ep-1"])

    assert "Status: superseded" in output
    assert "Superseded by: ep-2" in output
    assert "Old memory" in output


@pytest.mark.asyncio
async def test_memory_superseded_lists_hidden_records(monkeypatch):
    monkeypatch.setattr("components.commands.memory.QueryBasedAPIProxy", lambda **_: FakeAPI())
    command = Memory()
    command.plugin = FakePlugin()

    output = await _run(command, "superseded", [])

    assert "Superseded episodes" in output
    assert "ep-1" in output


@pytest.mark.asyncio
async def test_archive_and_restore_write_audit(monkeypatch):
    monkeypatch.setattr("components.commands.memory.QueryBasedAPIProxy", lambda **_: FakeAPI())
    command = Memory()
    plugin = FakePlugin()
    command.plugin = plugin

    archive_output = await _run(command, "archive", ["ep-1"])
    restore_output = await _run(command, "restore", ["ep-1"])

    assert "status set to archived" in archive_output
    assert "status set to active" in restore_output
    assert plugin.memory_store.updated == [("ep-1", "archived"), ("ep-1", "active")]
    assert [entry["operation"] for entry in plugin.memory_store.audit_entries] == [
        "archived",
        "restore",
    ]


@pytest.mark.asyncio
async def test_memory_export_l2_is_scoped_and_writes_audit(monkeypatch):
    monkeypatch.setattr("components.commands.memory.QueryBasedAPIProxy", lambda **_: FakeAPI())
    command = Memory()
    plugin = FakePlugin()
    command.plugin = plugin

    output = await _run(command, "export", ["l2"])
    data = json.loads(output)

    assert data["scope_key"] == "bot-1:group_1"
    assert data["user_key"] == "bot-1:group_1"
    assert data["count"] == 1
    assert data["episodes"][0]["id"] == "ep-1"
    assert plugin.memory_store.export_calls == [{
        "collection_id": "kb-1",
        "user_key": "bot-1:group_1",
        "include_statuses": ["active", "archived", "deleted", "superseded"],
    }]
    assert plugin.memory_store.audit_entries[-1]["operation"] == "export_l2"


@pytest.mark.asyncio
async def test_memory_import_l2_forces_current_scope_and_writes_audit(monkeypatch):
    monkeypatch.setattr("components.commands.memory.QueryBasedAPIProxy", lambda **_: FakeAPI())
    command = Memory()
    plugin = FakePlugin()
    command.plugin = plugin
    payload = json.dumps({
        "episodes": [
            {
                "id": "foreign-id",
                "content": "Imported memory",
                "user_key": "foreign-scope",
            }
        ]
    })

    output = await _run(command, "import", ["l2", payload])

    assert "Imported 1 L2 episode" in output
    assert plugin.memory_store.import_calls[0]["user_key"] == "bot-1:group_1"
    assert plugin.memory_store.import_calls[0]["episodes"][0]["user_key"] == "foreign-scope"
    assert plugin.memory_store.audit_entries[-1]["operation"] == "import_l2"


@pytest.mark.asyncio
async def test_memory_delete_l2_bulk_is_scoped_and_writes_audit(monkeypatch):
    monkeypatch.setattr("components.commands.memory.QueryBasedAPIProxy", lambda **_: FakeAPI())
    command = Memory()
    plugin = FakePlugin()
    command.plugin = plugin

    output = await _run(command, "delete", ["--status", "archived"])

    assert "Deleted 2 L2 episode" in output
    assert plugin.memory_store.delete_calls == [{
        "collection_id": "kb-1",
        "user_key": "bot-1:group_1",
        "sender_id": "",
        "tag": "",
        "before": "",
        "include_statuses": ["archived"],
    }]
    assert plugin.memory_store.audit_entries[-1]["operation"] == "delete_l2_bulk"
    assert plugin.memory_store.audit_entries[-1]["metadata"]["episode_ids"] == [
        "archived-1",
        "archived-2",
    ]
