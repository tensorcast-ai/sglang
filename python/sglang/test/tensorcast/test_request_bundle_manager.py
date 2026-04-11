# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Integration tests for the Tensorcast request-bundle manager."""

# ruff: noqa: E402

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import torch

from sglang.test.tensorcast.test_support import install_memory_pool_host_stub

install_memory_pool_host_stub()

from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    HydrateInstanceOpRequest,
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_publish import (
    RequestBundlePublishAggregator,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    RequestBundleStateError,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    PreparedBundleBindAction,
    PreparedBundleLifecycleState,
    PreparedHoldRef,
    PreparedHoldSetState,
    PreparedSlotToken,
    RankCoord,
    RankInstallState,
    PagePublicationState,
    RequestBundleLifecycleState,
)
from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
    TensorcastStore,
)
from sglang.test.tensorcast.test_tensorcast_store import (
    FakeHostKVCache,
    FakeTensorcastPageClient,
    build_storage_config,
)


class RequestBundleManagerTest(unittest.TestCase):
    def _install_prepared_bundle(
        self,
        store: TensorcastStore,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        prompt_token_ids: list[int],
    ) -> None:
        manager = store.request_bundle_manager
        manager.prepared_bundle_registry.begin_prepare(
            logical_request_id=logical_request_id,
            target_instance_id="instance-b",
            publish_manifest_digest=publish_manifest_digest,
            artifact_manifest_digest=f"artifact:{publish_manifest_digest}",
            engine_owned_manifest_sha256=f"engine:{publish_manifest_digest}",
            required_ranks=(RankCoord(tp_rank=0, pp_rank=0),),
            prompt_token_digest=get_hash_str(prompt_token_ids),
            cutoff_token_count=len(prompt_token_ids),
            now_ms=100,
        )
        manager.prepared_bundle_registry.mark_rank_install(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
            rank=RankCoord(tp_rank=0, pp_rank=0),
            state=RankInstallState.READY,
            hydrated_page_count=1,
            runnable_prefix_tokens=len(prompt_token_ids),
            local_install_handle=f"install:{publish_manifest_digest}",
        )
        manager.prepared_hold_registry.install_hold_set(
            hold_set_id=f"hold:{publish_manifest_digest}",
            refs=(
                PreparedHoldRef(
                    logical_page_index=0,
                    page_hash=f"page:{publish_manifest_digest}:0",
                    slot_token=PreparedSlotToken(slot_index=1, slot_generation=1),
                    artifact_id=f"artifact:{publish_manifest_digest}:0",
                ),
            ),
            now_ms=101,
        )
        manager.prepared_bundle_registry.mark_prepared(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
            now_ms=102,
            prepared_hold_set_id=f"hold:{publish_manifest_digest}",
        )

    def test_request_bundle_bind_attaches_then_consume_marks_bundle_consumed(
        self,
    ) -> None:
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(),
            FakeHostKVCache([1.0, 2.0]),
            page_client=client,
        )
        manager = store.request_bundle_manager
        self._install_prepared_bundle(
            store,
            logical_request_id="rid-bind",
            publish_manifest_digest="manifest-bind",
            prompt_token_ids=[101, 102],
        )

        bind_result = manager.bind_prepared_bundle_for_generate(
            logical_request_id="rid-bind",
            scheduler_rid="rid-bind",
            prompt_token_ids=[101, 102],
            requested_at_ms=200,
        )

        self.assertEqual(bind_result.action, PreparedBundleBindAction.ATTACHED)
        self.assertIsNotNone(bind_result.record)
        self.assertEqual(
            bind_result.record.state, PreparedBundleLifecycleState.ATTACHED
        )
        self.assertEqual(
            bind_result.prepared_bundle_key,
            "prepared::rid-bind::manifest-bind",
        )
        consumed_record = manager.consume_attached_prepared_bundle_for_generate(
            logical_request_id="rid-bind",
            publish_manifest_digest="manifest-bind",
            prompt_token_ids=[101, 102],
            requested_at_ms=220,
            install_prepared_bundle=lambda *_: None,
        )
        self.assertEqual(
            consumed_record.state,
            PreparedBundleLifecycleState.CONSUMED,
        )
        self.assertIsNone(consumed_record.prepared_hold_set_id)
        hold_set = manager.prepared_hold_registry.get("hold:manifest-bind")
        self.assertIsNotNone(hold_set)
        assert hold_set is not None
        self.assertEqual(hold_set.state, PreparedHoldSetState.RELEASED)

        manager.cleanup_bound_prepared_bundle(
            publish_manifest_digest="manifest-bind",
            requested_at_ms=300,
        )
        cleaned_record = manager.prepared_bundle_registry.get(
            logical_request_id="rid-bind",
            publish_manifest_digest="manifest-bind",
        )
        self.assertIsNotNone(cleaned_record)
        assert cleaned_record is not None
        self.assertEqual(cleaned_record.state, PreparedBundleLifecycleState.EVICTED)

    def test_request_bundle_bind_falls_back_on_prompt_digest_mismatch(self) -> None:
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(),
            FakeHostKVCache([1.0, 2.0]),
            page_client=client,
        )
        manager = store.request_bundle_manager
        self._install_prepared_bundle(
            store,
            logical_request_id="rid-fallback",
            publish_manifest_digest="manifest-fallback",
            prompt_token_ids=[201, 202],
        )

        bind_result = manager.bind_prepared_bundle_for_generate(
            logical_request_id="rid-fallback",
            scheduler_rid="rid-fallback",
            prompt_token_ids=[201, 999],
            requested_at_ms=200,
        )

        self.assertEqual(bind_result.action, PreparedBundleBindAction.FALLBACK)
        record = manager.prepared_bundle_registry.get(
            logical_request_id="rid-fallback",
            publish_manifest_digest="manifest-fallback",
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.state, PreparedBundleLifecycleState.PREPARED)

    def test_live_request_tracking_publishes_ready_local_rank_snapshot(self) -> None:
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([1.0, 2.0, 3.0, 4.0], page_size=2),
            page_client=client,
        )
        manager = store.request_bundle_manager
        rank = RankCoord(tp_rank=0, pp_rank=0)
        manager.configure_instance_ops_runtime(
            instance_id="instance-a",
            coordinator_epoch="epoch-1",
        )
        manager.start_live_request_tracking(
            logical_request_id="rid-live",
            engine_request_id="rid-live",
            prompt_token_ids=[101, 102, 103, 104],
            requested_at_ms=100,
        )
        manager.observe_live_request_progress(
            logical_request_id="rid-live",
            visible_prompt_token_count=4,
            emitted_decode_token_count=0,
            now_ms=110,
        )

        request_state = manager.request_bundle_registry.get("rid-live")
        self.assertIsNotNone(request_state)
        assert request_state is not None
        self.assertEqual(request_state.required_ranks, (rank,))

        pages = manager.page_publication_registry.snapshot_rank(
            logical_request_id="rid-live",
            rank=rank,
        )
        self.assertEqual(len(pages), 2)
        for page in pages:
            client.data[page.page_hash] = torch.ones((2,), dtype=torch.float32)

        publish_result = manager.publish_live_request(
            logical_request_id="rid-live",
            publish_op_id="publish-op-1",
            requested_at_ms=120,
        )

        self.assertEqual(
            publish_result.request_state.state,
            RequestBundleLifecycleState.PUBLISHED,
        )
        self.assertEqual(publish_result.cutoff.cutoff_token_count, 4)
        self.assertEqual(publish_result.publish_manifest.cutoff_token_count, 4)
        self.assertEqual(publish_result.publish_manifest.tail_valid_tokens, 0)
        self.assertEqual(
            len(publish_result.publish_manifest.artifact_manifest.entries), 2
        )

    def test_live_request_publish_allows_mid_decode_requests(self) -> None:
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([1.0, 2.0, 3.0, 4.0], page_size=2),
            page_client=client,
        )
        manager = store.request_bundle_manager
        rank = RankCoord(tp_rank=0, pp_rank=0)
        manager.start_live_request_tracking(
            logical_request_id="rid-mid-decode",
            engine_request_id="rid-mid-decode",
            prompt_token_ids=[111, 112, 113, 114],
            requested_at_ms=100,
        )
        manager.observe_live_request_progress(
            logical_request_id="rid-mid-decode",
            visible_prompt_token_count=4,
            emitted_decode_token_count=5,
            now_ms=110,
        )

        pages = manager.page_publication_registry.snapshot_rank(
            logical_request_id="rid-mid-decode",
            rank=rank,
        )
        self.assertEqual(len(pages), 2)
        for page in pages:
            client.data[page.page_hash] = torch.ones((2,), dtype=torch.float32)

        publish_result = manager.publish_live_request(
            logical_request_id="rid-mid-decode",
            publish_op_id="publish-op-mid-decode",
            requested_at_ms=120,
        )

        self.assertEqual(
            publish_result.request_state.state,
            RequestBundleLifecycleState.PUBLISHED,
        )
        self.assertEqual(publish_result.cutoff.cutoff_token_count, 4)
        self.assertEqual(publish_result.publish_manifest.tail_valid_tokens, 0)
        self.assertEqual(
            len(publish_result.publish_manifest.artifact_manifest.entries), 2
        )

    def test_live_request_publish_keeps_full_prompt_cutoff_and_records_tail(self) -> None:
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([1.0, 2.0, 3.0, 4.0], page_size=2),
            page_client=client,
        )
        manager = store.request_bundle_manager
        rank = RankCoord(tp_rank=0, pp_rank=0)
        manager.start_live_request_tracking(
            logical_request_id="rid-tail",
            engine_request_id="rid-tail",
            prompt_token_ids=[201, 202, 203],
            requested_at_ms=100,
        )
        manager.observe_live_request_progress(
            logical_request_id="rid-tail",
            visible_prompt_token_count=3,
            emitted_decode_token_count=0,
            now_ms=110,
        )

        pages = manager.page_publication_registry.snapshot_rank(
            logical_request_id="rid-tail",
            rank=rank,
        )
        self.assertEqual(len(pages), 2)
        client.data[pages[0].page_hash] = torch.ones((2,), dtype=torch.float32)

        publish_result = manager.publish_live_request(
            logical_request_id="rid-tail",
            publish_op_id="publish-op-tail",
            requested_at_ms=120,
        )

        self.assertEqual(publish_result.cutoff.cutoff_token_count, 3)
        self.assertEqual(publish_result.cutoff.page_aligned_token_count, 2)
        self.assertEqual(publish_result.publish_manifest.cutoff_token_count, 3)
        self.assertEqual(publish_result.publish_manifest.tail_valid_tokens, 1)
        self.assertEqual(
            len(publish_result.publish_manifest.artifact_manifest.entries), 1
        )

    def test_live_request_cleanup_retains_publishable_prompt_snapshot(self) -> None:
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([1.0, 2.0], page_size=2),
            page_client=client,
        )
        manager = store.request_bundle_manager
        rank = RankCoord(tp_rank=0, pp_rank=0)
        manager.start_live_request_tracking(
            logical_request_id="rid-clean",
            engine_request_id="rid-clean",
            prompt_token_ids=[301, 302],
            requested_at_ms=100,
        )
        manager.observe_live_request_progress(
            logical_request_id="rid-clean",
            visible_prompt_token_count=2,
            emitted_decode_token_count=0,
            now_ms=110,
        )

        self.assertIsNotNone(manager.request_bundle_registry.get("rid-clean"))
        self.assertEqual(
            len(
                manager.page_publication_registry.snapshot_rank(
                    logical_request_id="rid-clean",
                    rank=rank,
                )
            ),
            1,
        )

        manager.cleanup_live_request_tracking(
            logical_request_id="rid-clean",
            now_ms=120,
        )

        retained_state = manager.request_bundle_registry.get("rid-clean")
        self.assertIsNotNone(retained_state)
        assert retained_state is not None
        self.assertEqual(
            retained_state.state,
            RequestBundleLifecycleState.SOURCE_RETAINED,
        )
        self.assertIsNotNone(retained_state.retained_until_ms)
        self.assertEqual(
            len(
                manager.page_publication_registry.snapshot_rank(
                    logical_request_id="rid-clean",
                    rank=rank,
                )
            ),
            1,
        )

        page = manager.page_publication_registry.snapshot_rank(
            logical_request_id="rid-clean",
            rank=rank,
        )[0]
        client.data[page.page_hash] = torch.ones((2,), dtype=torch.float32)
        publish_result = manager.instance_publish_local(
            request=PublishInstanceOpRequest(
                logical_request_id="rid-clean",
                engine_request_id="rid-clean",
                publish_op_id="publish-op-retained",
                requested_cutoff_token_count=2,
                prompt_token_digest=get_hash_str([301, 302]),
                attention_arch="mha",
                dtype=str(store.dtype),
                page_size=2,
                emitted_decode_token_count=3,
                requested_at_ms=130,
            )
        )

        self.assertEqual(
            publish_result.request_state.state,
            RequestBundleLifecycleState.PUBLISHED,
        )
        self.assertEqual(publish_result.publish_manifest.cutoff_token_count, 2)

    def test_publish_waits_for_inflight_page_to_reach_ready(self) -> None:
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([1.0, 2.0], page_size=2),
            page_client=client,
        )
        manager = store.request_bundle_manager
        rank = RankCoord(tp_rank=0, pp_rank=0)
        prompt_token_ids = [601, 602]
        page_hash = get_hash_str(prompt_token_ids)
        manager.start_live_request_tracking(
            logical_request_id="rid-inflight",
            engine_request_id="rid-inflight",
            prompt_token_ids=prompt_token_ids,
            requested_at_ms=100,
        )
        manager.observe_live_request_progress(
            logical_request_id="rid-inflight",
            visible_prompt_token_count=2,
            emitted_decode_token_count=0,
            now_ms=110,
        )
        manager.page_publication_registry.set_page_state(
            logical_request_id="rid-inflight",
            rank=rank,
            logical_page_index=0,
            page_hash=page_hash,
            publication_state=PagePublicationState.INFLIGHT,
            updated_at_ms=115,
            host_resident=True,
        )

        def _complete_background_publication() -> None:
            time.sleep(0.05)
            client.data[page_hash] = torch.ones((2,), dtype=torch.float32)

        completion_thread = threading.Thread(
            target=_complete_background_publication,
            daemon=True,
        )
        completion_thread.start()
        publish_result = manager.publish_live_request(
            logical_request_id="rid-inflight",
            publish_op_id="publish-op-inflight",
            requested_at_ms=120,
        )
        completion_thread.join(timeout=1.0)

        self.assertEqual(
            publish_result.request_state.state,
            RequestBundleLifecycleState.PUBLISHED,
        )
        self.assertEqual(
            publish_result.publish_manifest.artifact_manifest.entries[0].artifact_id,
            client.artifact_id_for(page_hash),
        )

    def test_publish_pins_retained_source_window_while_waiting_for_closure(self) -> None:
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([1.0, 2.0], page_size=2),
            page_client=client,
        )
        manager = store.request_bundle_manager
        rank = RankCoord(tp_rank=0, pp_rank=0)
        prompt_token_ids = [701, 702]
        page_hash = get_hash_str(prompt_token_ids)
        manager.start_live_request_tracking(
            logical_request_id="rid-retained",
            engine_request_id="rid-retained",
            prompt_token_ids=prompt_token_ids,
            requested_at_ms=100,
        )
        manager.observe_live_request_progress(
            logical_request_id="rid-retained",
            visible_prompt_token_count=2,
            emitted_decode_token_count=0,
            now_ms=110,
        )
        manager.page_publication_registry.set_page_state(
            logical_request_id="rid-retained",
            rank=rank,
            logical_page_index=0,
            page_hash=page_hash,
            publication_state=PagePublicationState.INFLIGHT,
            updated_at_ms=115,
            host_resident=True,
        )
        manager.cleanup_live_request_tracking(
            logical_request_id="rid-retained",
            now_ms=120,
        )

        fake_elapsed_s = {"value": 0.0}
        publish_ready_at_ms = 61_200

        def _fake_monotonic() -> float:
            return fake_elapsed_s["value"]

        def _fake_sleep(seconds: float) -> None:
            fake_elapsed_s["value"] += float(seconds)

        def _fake_publish_ready(
            *,
            live_request,
            cutoff_token_count: int,
            now_ms: int,
        ) -> bool:
            _ = live_request, cutoff_token_count
            if now_ms >= publish_ready_at_ms:
                client.data[page_hash] = torch.ones((2,), dtype=torch.float32)
                manager.page_publication_registry.set_page_state(
                    logical_request_id="rid-retained",
                    rank=rank,
                    logical_page_index=0,
                    page_hash=page_hash,
                    publication_state=PagePublicationState.READY,
                    artifact_id=client.artifact_id_for(page_hash),
                    updated_at_ms=now_ms,
                    host_resident=True,
                )
                return True
            return False

        with (
            mock.patch(
                "sglang.srt.tensorcast.request_bundle.bundle_manager.time.monotonic",
                side_effect=_fake_monotonic,
            ),
            mock.patch(
                "sglang.srt.tensorcast.request_bundle.bundle_manager.time.sleep",
                side_effect=_fake_sleep,
            ),
            mock.patch.object(
                manager,
                "_publish_page_closure_ready_locked",
                side_effect=_fake_publish_ready,
            ),
        ):
            publish_result = manager.publish_live_request(
                logical_request_id="rid-retained",
                publish_op_id="publish-op-retained",
                requested_at_ms=130,
                timeout_ms=120_000,
            )

        self.assertEqual(
            publish_result.request_state.state,
            RequestBundleLifecycleState.PUBLISHED,
        )
        self.assertGreaterEqual(int(fake_elapsed_s["value"] * 1000.0), 61_000)

    def test_request_bundle_hydrate_local_accepts_group_publish_manifest(
        self,
    ) -> None:
        source_client = FakeTensorcastPageClient()
        prompt_token_ids = [701, 702, 703, 704]
        prompt_token_digest = get_hash_str(prompt_token_ids)

        source_rank0 = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([1.0, 2.0, 3.0, 4.0], page_size=2),
            page_client=source_client,
        )
        source_rank1 = TensorcastStore(
            build_storage_config(tp_rank=1),
            FakeHostKVCache([5.0, 6.0, 7.0, 8.0], page_size=2),
            page_client=source_client,
        )
        for store in (source_rank0, source_rank1):
            manager = store.request_bundle_manager
            manager.configure_instance_ops_runtime(
                instance_id="instance-a",
                coordinator_epoch="epoch-a",
            )
            manager.start_live_request_tracking(
                logical_request_id="rid-hydrate",
                engine_request_id="rid-hydrate",
                prompt_token_ids=prompt_token_ids,
                requested_at_ms=100,
            )
            manager.observe_live_request_progress(
                logical_request_id="rid-hydrate",
                visible_prompt_token_count=4,
                emitted_decode_token_count=0,
                now_ms=110,
            )
            pages = manager.page_publication_registry.snapshot_rank(
                logical_request_id="rid-hydrate",
                rank=RankCoord(tp_rank=store.local_rank, pp_rank=store.pp_rank),
            )
            for page in pages:
                source_client.data[page.page_hash] = torch.ones(
                    (2,), dtype=torch.float32
                )

        group_request = PublishInstanceOpRequest(
            logical_request_id="rid-hydrate",
            engine_request_id="rid-hydrate",
            publish_op_id="publish-op-hydrate",
            requested_cutoff_token_count=4,
            prompt_token_digest=prompt_token_digest,
            attention_arch="mha",
            dtype=str(source_rank0.dtype),
            page_size=2,
            requested_at_ms=120,
        )
        local_publish_rank0 = (
            source_rank0.request_bundle_manager.instance_publish_local(
                request=group_request
            )
        )
        local_publish_rank1 = (
            source_rank1.request_bundle_manager.instance_publish_local(
                request=group_request
            )
        )
        group_publish = RequestBundlePublishAggregator().aggregate(
            request=group_request,
            local_results=(local_publish_rank0, local_publish_rank1),
            now_ms=130,
            required_ranks=(
                RankCoord(tp_rank=0, pp_rank=0),
                RankCoord(tp_rank=1, pp_rank=0),
            ),
        )

        target_store = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([9.0, 10.0, 11.0, 12.0], page_size=2),
            page_client=source_client,
        )
        target_manager = target_store.request_bundle_manager
        target_manager.configure_instance_ops_runtime(
            instance_id="instance-b",
            coordinator_epoch="epoch-b",
        )

        hydrate_result = target_manager.instance_hydrate_local(
            request=HydrateInstanceOpRequest(
                logical_request_id="rid-hydrate",
                publish_manifest_digest=group_publish.publish_manifest.publish_manifest_digest,
                requested_at_ms=140,
            ),
            publish_manifest=group_publish.publish_manifest,
        )

        self.assertEqual(
            hydrate_result.prepared_bundle.state,
            PreparedBundleLifecycleState.PREPARED,
        )
        self.assertEqual(
            hydrate_result.prepared_bundle.required_ranks,
            (RankCoord(tp_rank=0, pp_rank=0),),
        )
        self.assertEqual(
            hydrate_result.prepared_bundle.rank_installs[0].state,
            RankInstallState.READY,
        )
        self.assertEqual(
            hydrate_result.prepared_bundle.prepared_hold_set_id,
            hydrate_result.hold_set.hold_set_id
            if hydrate_result.hold_set is not None
            else None,
        )
        self.assertIsNotNone(hydrate_result.hold_set)
        assert hydrate_result.hold_set is not None
        self.assertEqual(hydrate_result.hold_set.state, PreparedHoldSetState.ACTIVE)
        self.assertEqual(len(hydrate_result.hold_set.refs), 2)
        self.assertTrue(
            torch.equal(
                target_store.mem_pool_host.get_data_page(0),
                source_client.data[hydrate_result.hold_set.refs[0].page_hash],
            )
        )
        self.assertTrue(
            torch.equal(
                target_store.mem_pool_host.get_data_page(2),
                source_client.data[hydrate_result.hold_set.refs[1].page_hash],
            )
        )

    def test_request_bundle_cleanup_releases_unconsumed_hydrated_pages(self) -> None:
        source_client = FakeTensorcastPageClient()
        prompt_token_ids = [801, 802, 803, 804]
        prompt_token_digest = get_hash_str(prompt_token_ids)
        source_store = TensorcastStore(
            build_storage_config(tp_rank=0),
            FakeHostKVCache([1.0, 2.0, 3.0, 4.0], page_size=2),
            page_client=source_client,
        )
        source_manager = source_store.request_bundle_manager
        source_manager.configure_instance_ops_runtime(
            instance_id="instance-a",
            coordinator_epoch="epoch-a",
        )
        source_manager.start_live_request_tracking(
            logical_request_id="rid-cleanup",
            engine_request_id="rid-cleanup",
            prompt_token_ids=prompt_token_ids,
            requested_at_ms=100,
        )
        source_manager.observe_live_request_progress(
            logical_request_id="rid-cleanup",
            visible_prompt_token_count=4,
            emitted_decode_token_count=0,
            now_ms=110,
        )
        for page in source_manager.page_publication_registry.snapshot_rank(
            logical_request_id="rid-cleanup",
            rank=RankCoord(tp_rank=0, pp_rank=0),
        ):
            source_client.data[page.page_hash] = torch.ones((2,), dtype=torch.float32)

        publish_result = source_manager.instance_publish_local(
            request=PublishInstanceOpRequest(
                logical_request_id="rid-cleanup",
                engine_request_id="rid-cleanup",
                publish_op_id="publish-op-cleanup",
                requested_cutoff_token_count=4,
                prompt_token_digest=prompt_token_digest,
                attention_arch="mha",
                dtype=str(source_store.dtype),
                page_size=2,
                requested_at_ms=120,
            )
        )

        target_host_cache = FakeHostKVCache([9.0, 10.0, 11.0, 12.0], page_size=2)
        target_store = TensorcastStore(
            build_storage_config(tp_rank=0),
            target_host_cache,
            page_client=source_client,
        )
        target_manager = target_store.request_bundle_manager
        target_manager.configure_instance_ops_runtime(
            instance_id="instance-b",
            coordinator_epoch="epoch-b",
        )
        hydrate_result = target_manager.instance_hydrate_local(
            request=HydrateInstanceOpRequest(
                logical_request_id="rid-cleanup",
                publish_manifest_digest=publish_result.publish_manifest.publish_manifest_digest,
                requested_at_ms=140,
            ),
            publish_manifest=publish_result.publish_manifest,
        )
        self.assertIsNotNone(hydrate_result.hold_set)

        target_manager.cleanup_bound_prepared_bundle(
            publish_manifest_digest=publish_result.publish_manifest.publish_manifest_digest,
            requested_at_ms=160,
        )

        self.assertEqual(target_host_cache.free_calls, [[0, 2]])
        cleaned_record = target_manager.prepared_bundle_registry.get(
            logical_request_id="rid-cleanup",
            publish_manifest_digest=publish_result.publish_manifest.publish_manifest_digest,
        )
        self.assertIsNotNone(cleaned_record)
        assert cleaned_record is not None
        self.assertEqual(cleaned_record.state, PreparedBundleLifecycleState.EVICTED)

    def test_register_mem_pool_host_updates_manager_runtime(self) -> None:
        client = FakeTensorcastPageClient()
        original_host_cache = FakeHostKVCache([1.0, 2.0], page_size=2)
        store = TensorcastStore(
            build_storage_config(),
            original_host_cache,
            page_client=client,
        )
        replacement_host_cache = FakeHostKVCache([3.0, 4.0], page_size=2)
        store.register_mem_pool_host(replacement_host_cache)

        store.request_bundle_manager.release_prepared_hold_refs(
            (
                PreparedHoldRef(
                    logical_page_index=0,
                    page_hash="page:rebind:0",
                    slot_token=PreparedSlotToken(slot_index=0, slot_generation=100),
                    artifact_id="artifact:rebind:0",
                ),
            )
        )

        self.assertEqual(replacement_host_cache.free_calls, [[0]])
        self.assertEqual(original_host_cache.free_calls, [])


if __name__ == "__main__":
    unittest.main()
