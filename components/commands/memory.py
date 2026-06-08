from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator, NamedTuple

from langbot_plugin.api.definition.components.command.command import Command
from langbot_plugin.api.entities.builtin.command.context import (
    CommandReturn,
    ExecuteContext,
)
from langbot_plugin.api.proxies.query_based_api import QueryBasedAPIProxy

logger = logging.getLogger(__name__)


class _RuntimeContext(NamedTuple):
    api: QueryBasedAPIProxy
    bot_uuid: str
    query_vars: dict
    session_key: str
    user_key: str
    kb_id: str | None
    isolation: str
    config: dict


class Memory(Command):
    @staticmethod
    def _status_rank(status: str) -> int:
        return {"ERROR": 0, "WARN": 1, "OK": 2}.get(status, 1)

    @classmethod
    def _combine_status(cls, statuses: list[str]) -> str:
        if not statuses:
            return "WARN"
        return min(statuses, key=cls._status_rank)

    @staticmethod
    def _health_line(status: str, detail: str) -> str:
        return f"- {status}: {detail}"

    @staticmethod
    def _format_episode_detail(store, episode: dict) -> str:
        tags = ", ".join(episode.get("tags", [])) or "-"
        content = episode.get("content", "") or "-"
        lines = ["[Memory Episode]"]
        lines.append(f"ID: {episode.get('id', '-')}")
        lines.append(f"Status: {episode.get('status', 'active')}")
        lines.append(f"Timestamp: {episode.get('timestamp') or '-'}")
        lines.append(f"Importance: {episode.get('importance', 2)}")
        lines.append(f"Tags: {tags}")
        lines.append(f"Sender: {episode.get('sender_name') or '-'} ({episode.get('sender_id') or '-'})")
        lines.append(f"Source: {episode.get('source') or '-'}")
        lines.append(f"Superseded by: {episode.get('superseded_by') or '-'}")
        lines.append(f"Content: {store._preview_text(content, 500)}")
        return "\n".join(lines)

    @staticmethod
    def _parse_status_options(
        store,
        params: list[str],
    ) -> tuple[list[str] | None, list[str]]:
        include_statuses = [store.EPISODE_STATUS_ACTIVE]
        remaining: list[str] = []
        index = 0
        while index < len(params):
            value = params[index]
            if value == "--include-superseded":
                if store.EPISODE_STATUS_SUPERSEDED not in include_statuses:
                    include_statuses.append(store.EPISODE_STATUS_SUPERSEDED)
                index += 1
                continue
            if value == "--include-archived":
                if store.EPISODE_STATUS_ARCHIVED not in include_statuses:
                    include_statuses.append(store.EPISODE_STATUS_ARCHIVED)
                index += 1
                continue
            if value == "--include-deleted":
                if store.EPISODE_STATUS_DELETED not in include_statuses:
                    include_statuses.append(store.EPISODE_STATUS_DELETED)
                index += 1
                continue
            if value == "--all-statuses":
                include_statuses = sorted(store.EPISODE_STATUSES)
                index += 1
                continue
            if value == "--status":
                if index + 1 >= len(params):
                    raise ValueError(
                        "Usage: --status <active|superseded|archived|deleted>"
                    )
                status = str(params[index + 1]).strip().lower()
                if status not in store.EPISODE_STATUSES:
                    raise ValueError(
                        "status must be one of: active, superseded, archived, deleted"
                    )
                include_statuses = [status]
                index += 2
                continue
            remaining.append(value)
            index += 1
        return include_statuses, remaining

    @staticmethod
    async def _build_runtime_context(
        plugin,
        context: ExecuteContext,
    ) -> _RuntimeContext:
        store = plugin.memory_store
        api = QueryBasedAPIProxy(
            query_id=context.query_id,
            plugin_runtime_handler=plugin.plugin_runtime_handler,
        )
        bot_uuid = await api.get_bot_uuid()
        query_vars = await api.get_query_vars()
        session_key, user_key, kb_id, isolation, config = (
            await store.resolve_user_context(context.session, bot_uuid)
        )
        return _RuntimeContext(
            api=api,
            bot_uuid=bot_uuid,
            query_vars=query_vars,
            session_key=session_key,
            user_key=user_key,
            kb_id=kb_id,
            isolation=isolation,
            config=config,
        )

    @staticmethod
    async def _is_memory_kb_active(
        store,
        api: QueryBasedAPIProxy,
        kb_id: str | None,
    ) -> bool:
        if not kb_id:
            return False
        pipeline_kbs = await api.list_pipeline_knowledge_bases()
        return any(kb.get("uuid") == kb_id for kb in pipeline_kbs)

    @staticmethod
    async def _run_metadata_filter_probe(
        plugin,
        ctx: _RuntimeContext,
        embedding_model_uuid: str,
    ) -> tuple[str, list[str]]:
        probe_id = uuid.uuid4().hex[:10]
        id_a = f"ltm-health-{probe_id}-a"
        id_b = f"ltm-health-{probe_id}-b"
        user_a = f"ltm-health:{probe_id}:a"
        user_b = f"ltm-health:{probe_id}:b"
        probe_text_a = f"LongTermMemory health probe {probe_id} alpha"
        probe_text_b = f"LongTermMemory health probe {probe_id} beta"
        ids = [id_a, id_b]
        lines: list[str] = []
        status = "OK"

        def mark(new_status: str) -> None:
            nonlocal status
            status = Memory._combine_status([status, new_status])

        try:
            vectors = await plugin.invoke_embedding(
                embedding_model_uuid,
                [probe_text_a, probe_text_b],
            )
            await plugin.vector_upsert(
                collection_id=ctx.kb_id,
                vectors=vectors,
                ids=ids,
                metadata=[
                    {
                        "content": probe_text_a,
                        "user_key": user_a,
                        "source": "health_probe",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    {
                        "content": probe_text_b,
                        "user_key": user_b,
                        "source": "health_probe",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                ],
                documents=[probe_text_a, probe_text_b],
            )
            lines.append("OK: wrote temporary metadata probe records")

            search_results = await plugin.vector_search(
                collection_id=ctx.kb_id,
                query_vector=vectors[0],
                top_k=5,
                filters={"user_key": user_a},
            )
            search_ids = {item.get("id") for item in search_results}
            if id_b in search_ids:
                mark("ERROR")
                lines.append(
                    "ERROR: vector_search filter leaked another user_key result"
                )
            elif id_a in search_ids:
                lines.append("OK: vector_search respects user_key metadata filter")
            else:
                mark("WARN")
                lines.append(
                    "WARN: vector_search filter did not return its own probe record"
                )

            try:
                listed = await plugin.vector_list(
                    collection_id=ctx.kb_id,
                    filters={"user_key": user_a},
                    limit=10,
                    offset=0,
                )
                list_ids = {item.get("id") for item in listed.get("items", [])}
                if id_b in list_ids:
                    mark("ERROR")
                    lines.append(
                        "ERROR: vector_list filter leaked another user_key result"
                    )
                elif id_a in list_ids:
                    lines.append("OK: vector_list respects user_key metadata filter")
                else:
                    mark("WARN")
                    lines.append(
                        "WARN: vector_list did not return its own probe record"
                    )
            except Exception as exc:
                mark("WARN")
                lines.append(f"WARN: vector_list probe failed: {exc}")

            try:
                deleted = await plugin.vector_delete(
                    collection_id=ctx.kb_id,
                    filters={"user_key": user_a},
                )
                if deleted == 1:
                    lines.append("OK: vector_delete respects user_key metadata filter")
                elif deleted == 0:
                    mark("WARN")
                    lines.append(
                        "WARN: vector_delete filter did not delete its own probe record"
                    )
                else:
                    mark("ERROR")
                    lines.append(
                        "ERROR: vector_delete filter deleted more probe records than expected"
                    )
            except Exception as exc:
                mark("WARN")
                lines.append(f"WARN: vector_delete filter probe failed: {exc}")

        except Exception as exc:
            mark("ERROR")
            lines.append(f"ERROR: metadata filter probe failed: {exc}")
        finally:
            cleanup_lines = []
            for episode_id in ids:
                try:
                    await plugin.vector_delete(
                        collection_id=ctx.kb_id,
                        file_ids=[episode_id],
                    )
                    cleanup_lines.append(f"OK: cleaned probe record {episode_id}")
                except Exception as exc:
                    cleanup_lines.append(
                        f"WARN: failed to clean probe record {episode_id}: {exc}"
                    )
            lines.extend(cleanup_lines)

        return status, lines

    def __init__(self):
        super().__init__()

        @self.subcommand(
            name="",
            help="Show memory overview",
            usage="!memory",
            aliases=[],
        )
        async def root(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            ctx = await self._build_runtime_context(self.plugin, context)

            sender_id = str(ctx.query_vars.get("sender_id", "") or "")
            session_profile = await store.load_session_profile(ctx.session_key)
            speaker_profile = (
                await store.load_speaker_profile(ctx.session_key, sender_id)
                if sender_id
                else {}
            )
            kb_active = await self._is_memory_kb_active(store, ctx.api, ctx.kb_id)

            lines = [f"[Memory] mode: {ctx.isolation}"]
            lines.append(f"Session: {ctx.session_key}")
            lines.append(f"Key: {ctx.user_key}")
            lines.append(
                f"Current speaker: {ctx.query_vars.get('sender_name') or '-'} ({sender_id or '-'})"
            )

            if store.has_profile_data(session_profile):
                lines.append(
                    f"Session profile: name={session_profile.get('name', '-')}, "
                    f"{len(session_profile.get('traits', []))} traits, "
                    f"{len(session_profile.get('preferences', []))} prefs"
                )
            else:
                lines.append("Session profile: (empty)")

            if sender_id and store.has_profile_data(speaker_profile):
                lines.append(
                    f"Speaker profile: name={speaker_profile.get('name', '-')}, "
                    f"{len(speaker_profile.get('traits', []))} traits, "
                    f"{len(speaker_profile.get('preferences', []))} prefs"
                )
            elif sender_id:
                lines.append("Speaker profile: (empty)")

            if ctx.kb_id and kb_active:
                lines.append(f"L2 (Episodic): KB={ctx.kb_id[:12]}... active")
            elif ctx.kb_id:
                lines.append(f"L2 (Episodic): KB={ctx.kb_id[:12]}... configured but inactive in this pipeline")
            else:
                lines.append("L2 (Episodic): no KB configured")

            yield CommandReturn(text="\n".join(lines))

        @self.subcommand(
            name="health",
            help="Check memory KB configuration and metadata filter safety",
            usage="!memory health",
            aliases=["h"],
        )
        async def health_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            ctx = await self._build_runtime_context(self.plugin, context)
            statuses: list[str] = []
            lines = ["[Memory Health]"]

            if ctx.kb_id:
                statuses.append("OK")
                lines.append(self._health_line("OK", f"memory KB configured: {ctx.kb_id}"))
            else:
                statuses.append("ERROR")
                lines.append(self._health_line("ERROR", "no memory KB configured"))

            kb_active = await self._is_memory_kb_active(
                self.plugin.memory_store,
                ctx.api,
                ctx.kb_id,
            )
            if kb_active:
                statuses.append("OK")
                lines.append(self._health_line("OK", "memory KB is active in this pipeline"))
            else:
                statuses.append("ERROR")
                lines.append(
                    self._health_line(
                        "ERROR",
                        "memory KB is not active in the current pipeline",
                    )
                )

            embedding_model_uuid = str(ctx.config.get("embedding_model_uuid", "") or "")
            if embedding_model_uuid:
                statuses.append("OK")
                lines.append(
                    self._health_line(
                        "OK",
                        f"embedding model configured: {embedding_model_uuid}",
                    )
                )
            else:
                statuses.append("ERROR")
                lines.append(self._health_line("ERROR", "no embedding model configured"))

            if ctx.kb_id and kb_active and embedding_model_uuid:
                probe_status, probe_lines = await self._run_metadata_filter_probe(
                    self.plugin,
                    ctx,
                    embedding_model_uuid,
                )
                statuses.append(probe_status)
                lines.extend(f"- {line}" for line in probe_lines)
            else:
                statuses.append("WARN")
                lines.append(
                    self._health_line(
                        "WARN",
                        "metadata filter probe skipped because prerequisites failed",
                    )
                )

            final_status = self._combine_status(statuses)
            if final_status == "ERROR":
                lines.append(
                    "Result: ERROR - LongTermMemory is not safe for scoped recall in this pipeline yet."
                )
            elif final_status == "WARN":
                lines.append(
                    "Result: WARN - LongTermMemory can run, but some safety checks need attention."
                )
            else:
                lines.append(
                    "Result: OK - memory KB configuration and scoped metadata filters look usable."
                )

            yield CommandReturn(text="\n".join(lines))

        @self.subcommand(
            name="profile",
            help="Show session and speaker profiles",
            usage="!memory profile",
            aliases=["p"],
        )
        async def profile_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            ctx = await self._build_runtime_context(self.plugin, context)

            sender_id = str(ctx.query_vars.get("sender_id", "") or "")
            session_profile = await store.load_session_profile(ctx.session_key)

            lines = ["[Session Profile]"]
            lines.append(f"Name: {session_profile.get('name') or '(not set)'}")
            lines.append(
                f"Traits: {', '.join(session_profile.get('traits', [])) or '(none)'}"
            )
            lines.append(
                "Preferences: "
                f"{', '.join(session_profile.get('preferences', [])) or '(none)'}"
            )
            lines.append(f"Notes: {session_profile.get('notes') or '(none)'}")

            if sender_id:
                speaker_profile = await store.load_speaker_profile(ctx.session_key, sender_id)
                lines.append("")
                lines.append("[Current Speaker Profile]")
                lines.append(f"Name: {speaker_profile.get('name') or '(not set)'}")
                lines.append(
                    f"Traits: {', '.join(speaker_profile.get('traits', [])) or '(none)'}"
                )
                lines.append(
                    "Preferences: "
                    f"{', '.join(speaker_profile.get('preferences', [])) or '(none)'}"
                )
                lines.append(f"Notes: {speaker_profile.get('notes') or '(none)'}")

            yield CommandReturn(text="\n".join(lines))

        @self.subcommand(
            name="search",
            help="Search episodic memories",
            usage="!memory search [--include-superseded] [--include-archived] [--status <status>] <query>",
            aliases=["s"],
        )
        async def search_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            ctx = await self._build_runtime_context(self.plugin, context)

            if not ctx.kb_id or not await self._is_memory_kb_active(store, ctx.api, ctx.kb_id):
                yield CommandReturn(
                    text="[Memory] Memory knowledge base is not configured for the current pipeline."
                )
                return

            embedding_model_uuid = ctx.config.get("embedding_model_uuid", "")

            try:
                include_statuses, query_parts = self._parse_status_options(
                    store,
                    context.crt_params,
                )
            except ValueError as exc:
                yield CommandReturn(text=str(exc))
                return

            if not query_parts:
                yield CommandReturn(text="Usage: !memory search <query>")
                return

            query = " ".join(query_parts)

            episodes = await store.search_episodes(
                collection_id=ctx.kb_id,
                embedding_model_uuid=embedding_model_uuid,
                query=query,
                user_key=ctx.user_key,
                top_k=10,
                include_statuses=include_statuses,
            )

            if not episodes:
                yield CommandReturn(text="[Memory] No episodes found.")
                return

            lines = [f"[Memory] Found {len(episodes)} episode(s):"]
            for ep in episodes:
                ts = ep["timestamp"][:10] if ep.get("timestamp") else "?"
                imp = ep.get("importance", 2)
                status = ep.get("status", "active")
                tags = ", ".join(ep.get("tags", []))
                tag_str = f" [{tags}]" if tags else ""
                eid = ep.get("id", "?")
                lines.append(
                    f"  [{eid}] {ts} ({status}, imp:{imp}){tag_str} {ep['content']}"
                )

            yield CommandReturn(text="\n".join(lines))

        @self.subcommand(
            name="list",
            help="List episodic memories with pagination",
            usage="!memory list [--include-superseded] [--include-archived] [--status <status>] [page]",
            aliases=["l", "ls"],
        )
        async def list_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            ctx = await self._build_runtime_context(self.plugin, context)

            if not ctx.kb_id or not await self._is_memory_kb_active(store, ctx.api, ctx.kb_id):
                yield CommandReturn(
                    text="[Memory] Memory knowledge base is not configured for the current pipeline."
                )
                return

            page = 1
            try:
                include_statuses, remaining = self._parse_status_options(
                    store,
                    context.crt_params,
                )
            except ValueError as exc:
                yield CommandReturn(text=str(exc))
                return

            if remaining:
                try:
                    page = max(1, int(remaining[0]))
                except ValueError:
                    yield CommandReturn(text="Usage: !memory list [page]")
                    return
                if len(remaining) > 1:
                    yield CommandReturn(text="Usage: !memory list [page]")
                    return

            page_size = 10
            offset = (page - 1) * page_size

            episodes, total = await store.list_episodes(
                collection_id=ctx.kb_id,
                user_key=ctx.user_key,
                limit=page_size,
                offset=offset,
                include_statuses=include_statuses,
            )

            if not episodes:
                yield CommandReturn(text="[Memory] No episodes found.")
                return

            total_str = str(total) if total >= 0 else "?"
            lines = [f"[Memory] Episodes (page {page}, {total_str} total):"]
            for ep in episodes:
                ts = ep["timestamp"][:10] if ep.get("timestamp") else "?"
                imp = ep.get("importance", 2)
                status = ep.get("status", "active")
                tags = ", ".join(ep.get("tags", []))
                tag_str = f" [{tags}]" if tags else ""
                eid = ep.get("id", "?")
                content_preview = store._preview_text(ep.get("content", ""), 80)
                lines.append(
                    f"  [{eid}] {ts} ({status}, imp:{imp}){tag_str} {content_preview}"
                )

            yield CommandReturn(text="\n".join(lines))

        @self.subcommand(
            name="forget",
            help="Delete an episode by ID",
            usage="!memory forget <episode_id>",
            aliases=["f", "del"],
        )
        async def forget_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            ctx = await self._build_runtime_context(self.plugin, context)

            if not ctx.kb_id or not await self._is_memory_kb_active(store, ctx.api, ctx.kb_id):
                yield CommandReturn(
                    text="[Memory] Memory knowledge base is not configured for the current pipeline."
                )
                return

            if not context.crt_params:
                yield CommandReturn(text="Usage: !memory forget <episode_id>")
                return

            episode_id = context.crt_params[0].strip()
            deleted = await store.delete_episode_by_id(
                collection_id=ctx.kb_id,
                episode_id=episode_id,
                user_key=ctx.user_key,
            )
            await store.append_audit_entry(
                scope_key=ctx.session_key,
                user_key=ctx.user_key,
                operation="forget",
                target_type="episode",
                target_id=episode_id,
                summary=f"Command deleted episode {episode_id}",
                sender_id=str(ctx.query_vars.get("sender_id", "") or ""),
                sender_name=str(ctx.query_vars.get("sender_name", "") or ""),
                query_id=context.query_id,
                metadata={"kb_id": ctx.kb_id, "deleted": deleted},
            )
            yield CommandReturn(text=f"[Memory] Episode {episode_id} deleted.")

        @self.subcommand(
            name="show",
            help="Show an episodic memory by ID",
            usage="!memory show <episode_id>",
            aliases=[],
        )
        async def show_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            ctx = await self._build_runtime_context(self.plugin, context)

            if not ctx.kb_id or not await self._is_memory_kb_active(store, ctx.api, ctx.kb_id):
                yield CommandReturn(
                    text="[Memory] Memory knowledge base is not configured for the current pipeline."
                )
                return

            if not context.crt_params:
                yield CommandReturn(text="Usage: !memory show <episode_id>")
                return

            episode_id = context.crt_params[0].strip()
            episode = await store.get_episode_by_id(
                collection_id=ctx.kb_id,
                episode_id=episode_id,
                user_key=ctx.user_key,
            )
            if not episode:
                yield CommandReturn(text=f"[Memory] Episode {episode_id} not found.")
                return
            yield CommandReturn(text=self._format_episode_detail(store, episode))

        @self.subcommand(
            name="superseded",
            help="List superseded episodic memories",
            usage="!memory superseded [page]",
            aliases=[],
        )
        async def superseded_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            async for item in self._list_by_status(
                context,
                self.plugin.memory_store.EPISODE_STATUS_SUPERSEDED,
            ):
                yield item

        @self.subcommand(
            name="archived",
            help="List archived episodic memories",
            usage="!memory archived [page]",
            aliases=[],
        )
        async def archived_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            async for item in self._list_by_status(
                context,
                self.plugin.memory_store.EPISODE_STATUS_ARCHIVED,
            ):
                yield item

        @self.subcommand(
            name="archive",
            help="Archive an episodic memory by ID",
            usage="!memory archive <episode_id>",
            aliases=[],
        )
        async def archive_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            async for item in self._set_episode_status(
                context,
                self.plugin.memory_store.EPISODE_STATUS_ARCHIVED,
            ):
                yield item

        @self.subcommand(
            name="restore",
            help="Restore an episodic memory to active status",
            usage="!memory restore <episode_id>",
            aliases=[],
        )
        async def restore_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            async for item in self._set_episode_status(
                context,
                self.plugin.memory_store.EPISODE_STATUS_ACTIVE,
            ):
                yield item

        @self.subcommand(
            name="audit",
            help="List or export scoped memory audit entries",
            usage="!memory audit [page|export]",
            aliases=[],
        )
        async def audit_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            ctx = await self._build_runtime_context(self.plugin, context)

            if context.crt_params and context.crt_params[0] == "export":
                entries = await store.export_audit_entries(ctx.session_key)
                export_data = {
                    "version": 1,
                    "exported_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),
                    "scope_key": ctx.session_key,
                    "user_key": ctx.user_key,
                    "entries": entries,
                }
                await store.append_audit_entry(
                    scope_key=ctx.session_key,
                    user_key=ctx.user_key,
                    operation="audit_export",
                    target_type="audit",
                    target_id=ctx.session_key,
                    summary=f"Exported {len(entries)} audit entries",
                    sender_id=str(ctx.query_vars.get("sender_id", "") or ""),
                    sender_name=str(ctx.query_vars.get("sender_name", "") or ""),
                    query_id=context.query_id,
                )
                yield CommandReturn(
                    text=json.dumps(export_data, ensure_ascii=False, indent=2)
                )
                return

            page = 1
            if context.crt_params:
                try:
                    page = max(1, int(context.crt_params[0]))
                except ValueError:
                    yield CommandReturn(text="Usage: !memory audit [page|export]")
                    return

            page_size = 10
            offset = (page - 1) * page_size
            entries, total = await store.list_audit_entries(
                ctx.session_key,
                limit=page_size,
                offset=offset,
            )
            if not entries:
                yield CommandReturn(text="[Memory] No audit entries found.")
                return

            lines = [f"[Memory Audit] page {page}, {total} total"]
            for entry in entries:
                ts = str(entry.get("timestamp", ""))[:19] or "?"
                operation = entry.get("operation", "-")
                target = entry.get("target_id", "-")
                summary = entry.get("summary", "")
                lines.append(f"  {ts} {operation} {target}: {summary}")
            yield CommandReturn(text="\n".join(lines))

        @self.subcommand(
            name="export",
            help="Export L1 profiles for the current session",
            usage="!memory export",
            aliases=["e"],
        )
        async def export_cmd(
            self: Memory,
            context: ExecuteContext,
        ) -> AsyncGenerator[CommandReturn, None]:
            store = self.plugin.memory_store
            ctx = await self._build_runtime_context(self.plugin, context)

            profiles = await store.export_profiles_by_scope(ctx.session_key)

            if not profiles:
                yield CommandReturn(text="[Memory] No profiles to export.")
                return

            export_data = {
                "version": 1,
                "exported_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "profiles": profiles,
            }
            await store.append_audit_entry(
                scope_key=ctx.session_key,
                user_key=ctx.user_key,
                operation="export_profiles",
                target_type="profile",
                target_id=ctx.session_key,
                summary=f"Exported {len(profiles)} L1 profile entries",
                sender_id=str(ctx.query_vars.get("sender_id", "") or ""),
                sender_name=str(ctx.query_vars.get("sender_name", "") or ""),
                query_id=context.query_id,
            )

            yield CommandReturn(
                text=json.dumps(export_data, ensure_ascii=False, indent=2)
            )

    async def _list_by_status(
        self,
        context: ExecuteContext,
        status: str,
    ) -> AsyncGenerator[CommandReturn, None]:
        store = self.plugin.memory_store
        ctx = await self._build_runtime_context(self.plugin, context)

        if not ctx.kb_id or not await self._is_memory_kb_active(store, ctx.api, ctx.kb_id):
            yield CommandReturn(
                text="[Memory] Memory knowledge base is not configured for the current pipeline."
            )
            return

        page = 1
        if context.crt_params:
            try:
                page = max(1, int(context.crt_params[0]))
            except ValueError:
                yield CommandReturn(text=f"Usage: !memory {status} [page]")
                return

        page_size = 10
        offset = (page - 1) * page_size
        episodes, total = await store.list_episodes(
            collection_id=ctx.kb_id,
            user_key=ctx.user_key,
            limit=page_size,
            offset=offset,
            include_statuses=[status],
        )
        if not episodes:
            yield CommandReturn(text=f"[Memory] No {status} episodes found.")
            return

        total_str = str(total) if total >= 0 else "?"
        lines = [f"[Memory] {status.capitalize()} episodes (page {page}, {total_str} total):"]
        for ep in episodes:
            ts = ep["timestamp"][:10] if ep.get("timestamp") else "?"
            tags = ", ".join(ep.get("tags", []))
            tag_str = f" [{tags}]" if tags else ""
            eid = ep.get("id", "?")
            content_preview = store._preview_text(ep.get("content", ""), 80)
            lines.append(f"  [{eid}] {ts}{tag_str} {content_preview}")
        yield CommandReturn(text="\n".join(lines))

    async def _set_episode_status(
        self,
        context: ExecuteContext,
        status: str,
    ) -> AsyncGenerator[CommandReturn, None]:
        store = self.plugin.memory_store
        ctx = await self._build_runtime_context(self.plugin, context)

        if not ctx.kb_id or not await self._is_memory_kb_active(store, ctx.api, ctx.kb_id):
            yield CommandReturn(
                text="[Memory] Memory knowledge base is not configured for the current pipeline."
            )
            return

        embedding_model_uuid = str(ctx.config.get("embedding_model_uuid", "") or "")
        if not embedding_model_uuid:
            yield CommandReturn(text="[Memory] No embedding model configured.")
            return

        if not context.crt_params:
            yield CommandReturn(text=f"Usage: !memory {status} <episode_id>")
            return

        episode_id = context.crt_params[0].strip()
        episode = await store.update_episode_status(
            collection_id=ctx.kb_id,
            embedding_model_uuid=embedding_model_uuid,
            episode_id=episode_id,
            user_key=ctx.user_key,
            status=status,
        )
        if not episode:
            yield CommandReturn(text=f"[Memory] Episode {episode_id} not found.")
            return

        operation = "restore" if status == store.EPISODE_STATUS_ACTIVE else status
        await store.append_audit_entry(
            scope_key=ctx.session_key,
            user_key=ctx.user_key,
            operation=operation,
            target_type="episode",
            target_id=episode_id,
            summary=f"Set episode {episode_id} status to {status}",
            sender_id=str(ctx.query_vars.get("sender_id", "") or ""),
            sender_name=str(ctx.query_vars.get("sender_name", "") or ""),
            query_id=context.query_id,
            metadata={"kb_id": ctx.kb_id, "status": status},
        )
        yield CommandReturn(text=f"[Memory] Episode {episode_id} status set to {status}.")
