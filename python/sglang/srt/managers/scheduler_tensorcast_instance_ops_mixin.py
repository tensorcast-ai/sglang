from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
import os
import time
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.tensorcast.instance_ops.instance_ops_runtime import (
    InstanceOpsRuntimeCoordinator,
)
from sglang.srt.tensorcast.instance_ops.instance_directory import (
    TensorcastInstanceDirectoryConfig,
    TensorcastInstanceDirectoryRegistration,
    resolve_instance_agent_execution_endpoint,
)
from sglang.srt.tensorcast.request_bundle.bundle_manager import (
    RequestBundleManager,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    RequestBundleStateError,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    EvictLocalInstanceOpReqInput,
    EvictLocalInstanceOpRequest,
    EvictLocalInstanceOpResult,
    EvictLocalInstanceOpRespOutput,
    HydrateInstanceOpReqInput,
    HydrateInstanceOpRequest,
    HydrateInstanceOpRespOutput,
    InstanceOpStatus,
    PublishInstanceOpReqInput,
    PublishInstanceOpRequest,
    PublishInstanceOpRespOutput,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    HydratePreparedResult,
    PreparedBundleBindAction,
    PreparedBundleLifecycleState,
    PreparedBundleRecord,
    PublishManifestRecord,
    RankCoord,
    SourcePublishClosureResult,
)
from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
    TensorcastStore,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class _SchedulerTensorcastInstanceReqState:
    prepared_bundle_record: PreparedBundleRecord | None = None
    prepared_bundle_key: str | None = None
    prepared_bundle_claim_token: str | None = None
    prepared_bundle_host_node: object | None = None
    live_source_request_tracked: bool = False


class SchedulerTensorcastInstanceOpsMixin:
    def init_tensorcast_instance_ops(self: Scheduler) -> None:
        self._tensorcast_instance_ops_runtime: InstanceOpsRuntimeCoordinator | None = (
            None
        )
        self._tensorcast_instance_directory_registration: (
            TensorcastInstanceDirectoryRegistration | None
        ) = None
        self._tensorcast_instance_req_state: WeakKeyDictionary[
            Req, _SchedulerTensorcastInstanceReqState
        ] = WeakKeyDictionary()

    def _tensorcast_instance_state_for_req(
        self: Scheduler,
        req: Req,
        *,
        create: bool = False,
    ) -> _SchedulerTensorcastInstanceReqState | None:
        state = self._tensorcast_instance_req_state.get(req)
        if state is not None or not create:
            return state
        state = _SchedulerTensorcastInstanceReqState()
        self._tensorcast_instance_req_state[req] = state
        return state

    def _maybe_drop_tensorcast_instance_req_state(
        self: Scheduler,
        req: Req,
        state: _SchedulerTensorcastInstanceReqState | None = None,
    ) -> None:
        resolved_state = (
            state if state is not None else self._tensorcast_instance_req_state.get(req)
        )
        if resolved_state is None:
            return
        if (
            resolved_state.prepared_bundle_record is not None
            or resolved_state.prepared_bundle_key is not None
            or resolved_state.prepared_bundle_claim_token is not None
            or resolved_state.prepared_bundle_host_node is not None
            or resolved_state.live_source_request_tracked
        ):
            return
        self._tensorcast_instance_req_state.pop(req, None)

    def _tensorcast_storage_backend(self: Scheduler) -> TensorcastStore | None:
        if not self.enable_hicache_storage:
            return None
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller is None:
            return None
        storage_backend = getattr(cache_controller, "storage_backend", None)
        if not isinstance(storage_backend, TensorcastStore):
            return None
        return storage_backend

    def _tensorcast_instance_rank_coord(self: Scheduler) -> RankCoord:
        return RankCoord(tp_rank=self.tp_rank, pp_rank=self.pp_rank)

    def _tensorcast_request_bundle_manager(
        self: Scheduler,
    ) -> RequestBundleManager | None:
        storage_backend = self._tensorcast_storage_backend()
        if storage_backend is None:
            return None
        return storage_backend.request_bundle_manager

    def _tensorcast_instance_required_ranks(self: Scheduler) -> tuple[RankCoord, ...]:
        if self.dp_size != 1:
            raise RequestBundleStateError(
                "instance ops runtime fanout is not implemented for dp_size > 1"
            )
        return tuple(
            RankCoord(tp_rank=tp_rank, pp_rank=pp_rank)
            for pp_rank in range(self.pp_size)
            for tp_rank in range(self.tp_size)
        )

    def _require_tensorcast_instance_ops_runtime(
        self: Scheduler,
    ) -> InstanceOpsRuntimeCoordinator:
        if self._tensorcast_instance_ops_runtime is None:
            raise RequestBundleStateError(
                "Tensorcast instance ops runtime is not configured for this scheduler"
            )
        return self._tensorcast_instance_ops_runtime

    def _configure_tensorcast_instance_ops_runtime(self: Scheduler) -> None:
        storage_backend = self._tensorcast_storage_backend()
        manager = self._tensorcast_request_bundle_manager()
        if storage_backend is None or manager is None:
            return
        instance_id = f"{self.server_args.host}:{self.server_args.port}"
        try:
            coordinator_epoch_seed = (
                f"{instance_id}:pid-{os.getpid()}:ts-{int(time.time() * 1000)}"
                if self.world_group.rank_in_group == 0
                else None
            )
            coordinator_epoch = self.world_group.broadcast_object(
                coordinator_epoch_seed,
                src=0,
            )
            manager.configure_instance_ops_runtime(
                instance_id=instance_id,
                coordinator_epoch=coordinator_epoch,
            )
            if self.dp_size == 1:
                runtime = InstanceOpsRuntimeCoordinator(
                    current_rank=self._tensorcast_instance_rank_coord(),
                    object_group=self.world_group,
                )
                runtime.configure(
                    instance_id=instance_id,
                    coordinator_epoch=coordinator_epoch,
                    required_ranks=self._tensorcast_instance_required_ranks(),
                    now_ms=int(time.time() * 1000),
                )
                self._tensorcast_instance_ops_runtime = runtime
                self._maybe_start_tensorcast_instance_directory_registration(
                    storage_backend=storage_backend,
                    instance_id=instance_id,
                )
            else:
                logger.warning(
                    "Tensorcast instance ops runtime fanout is disabled because dp_size=%s is not supported yet",
                    self.dp_size,
                )
        except Exception:
            logger.exception("Tensorcast instance ops runtime configuration failed")

    def _build_tensorcast_instance_directory_registration(
        self: Scheduler,
        *,
        storage_backend: TensorcastStore,
        instance_id: str,
        execution_endpoint: str,
    ) -> TensorcastInstanceDirectoryRegistration:
        config = storage_backend.tensorcast_config
        return TensorcastInstanceDirectoryRegistration(
            TensorcastInstanceDirectoryConfig(
                global_store_address=config.instance_directory_address,
                daemon_address=config.daemon_address,
                instance_id=instance_id,
                engine=config.engine,
                execution_endpoint=execution_endpoint,
                signals_endpoint=config.instance_signals_endpoint,
                heartbeat_interval_ms=config.instance_directory_heartbeat_interval_ms,
                labels={"engine": config.engine},
            )
        )

    def _maybe_start_tensorcast_instance_directory_registration(
        self: Scheduler,
        *,
        storage_backend: TensorcastStore,
        instance_id: str,
    ) -> None:
        if self.world_group.rank_in_group != 0:
            return
        config = storage_backend.tensorcast_config
        global_store_address = str(config.instance_directory_address).strip()
        if not global_store_address:
            return
        execution_endpoint = resolve_instance_agent_execution_endpoint(
            listen_host=self.server_args.host,
            listen_port=self.server_args.port,
            configured_endpoint=config.instance_agent_execution_endpoint,
        )
        if execution_endpoint is None:
            logger.warning(
                "Tensorcast instance-directory registration skipped because server host=%s is wildcard and no explicit execution endpoint was configured",
                self.server_args.host,
            )
            return
        if self._tensorcast_instance_directory_registration is not None:
            self._tensorcast_instance_directory_registration.stop()
            self._tensorcast_instance_directory_registration = None
        registration = self._build_tensorcast_instance_directory_registration(
            storage_backend=storage_backend,
            instance_id=instance_id,
            execution_endpoint=execution_endpoint,
        )
        registration.start()
        self._tensorcast_instance_directory_registration = registration

    def _shutdown_tensorcast_instance_ops_runtime(self: Scheduler) -> None:
        registration = self._tensorcast_instance_directory_registration
        self._tensorcast_instance_directory_registration = None
        if registration is not None:
            registration.stop()
        self._tensorcast_instance_ops_runtime = None

    def tensorcast_instance_op_dispatch_entries(self: Scheduler) -> list[tuple[type, object]]:
        return [
            (
                PublishInstanceOpReqInput,
                self.handle_tensorcast_publish_instance_op_request,
            ),
            (
                HydrateInstanceOpReqInput,
                self.handle_tensorcast_hydrate_instance_op_request,
            ),
            (
                EvictLocalInstanceOpReqInput,
                self.handle_tensorcast_evict_local_instance_op_request,
            ),
        ]

    def _dispatch_output_uses_instance_ops_channel(
        self: Scheduler,
        output: object,
    ) -> bool:
        return isinstance(
            output,
            (
                PublishInstanceOpRespOutput,
                HydrateInstanceOpRespOutput,
                EvictLocalInstanceOpRespOutput,
            ),
        )

    def handle_tensorcast_publish_instance_op_request(
        self: Scheduler,
        recv_req: PublishInstanceOpReqInput,
    ) -> PublishInstanceOpRespOutput:
        try:
            return PublishInstanceOpRespOutput(
                status=InstanceOpStatus.SUCCESS,
                result=self.tensorcast_instance_publish_engine_request_id(
                    engine_request_id=recv_req.engine_request_id,
                    ttl_ms=recv_req.ttl_ms,
                    timeout_ms=recv_req.timeout_ms,
                    requested_cutoff_token_count=recv_req.requested_cutoff_token_count,
                    requested_at_ms=recv_req.requested_at_ms,
                    publish_op_id=recv_req.publish_op_id,
                ),
            )
        except Exception as exc:
            logger.exception(
                "Tensorcast instance publish request failed logical_request_id=%s publish_op_id=%s",
                recv_req.engine_request_id,
                recv_req.publish_op_id,
            )
            return PublishInstanceOpRespOutput(
                status=InstanceOpStatus.FAILED,
                error_message=str(exc),
            )

    def handle_tensorcast_hydrate_instance_op_request(
        self: Scheduler,
        recv_req: HydrateInstanceOpReqInput,
    ) -> HydrateInstanceOpRespOutput:
        try:
            logger.debug(
                "Tensorcast hydrate request received tp_rank=%s pp_rank=%s logical_request_id=%s manifest=%s",
                self.tp_rank,
                self.pp_rank,
                recv_req.request.logical_request_id,
                recv_req.request.publish_manifest_digest,
            )
            return HydrateInstanceOpRespOutput(
                status=InstanceOpStatus.SUCCESS,
                result=self.tensorcast_instance_hydrate(
                    request=recv_req.request,
                    publish_manifest=recv_req.publish_manifest,
                ),
            )
        except Exception as exc:
            logger.exception(
                "Tensorcast instance hydrate request failed logical_request_id=%s manifest=%s",
                recv_req.request.logical_request_id,
                recv_req.request.publish_manifest_digest,
            )
            return HydrateInstanceOpRespOutput(
                status=InstanceOpStatus.FAILED,
                error_message=str(exc),
            )

    def handle_tensorcast_evict_local_instance_op_request(
        self: Scheduler,
        recv_req: EvictLocalInstanceOpReqInput,
    ) -> EvictLocalInstanceOpRespOutput:
        try:
            return EvictLocalInstanceOpRespOutput(
                status=InstanceOpStatus.SUCCESS,
                result=self.tensorcast_instance_evict_local(recv_req.request),
            )
        except Exception as exc:
            logger.exception(
                "Tensorcast instance evict_local request failed logical_request_id=%s manifest=%s",
                recv_req.request.logical_request_id,
                recv_req.request.publish_manifest_digest,
            )
            return EvictLocalInstanceOpRespOutput(
                status=InstanceOpStatus.FAILED,
                error_message=str(exc),
            )

    def tensorcast_instance_publish(
        self: Scheduler,
        request: PublishInstanceOpRequest | dict,
    ) -> SourcePublishClosureResult:
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            raise RequestBundleStateError(
                "Tensorcast instance publish requires the Tensorcast HiCache backend"
            )
        typed_request = PublishInstanceOpRequest.model_validate(request)
        return self._require_tensorcast_instance_ops_runtime().publish(
            request=typed_request,
            execute_local=lambda resolved_request: manager.instance_publish_local(
                request=resolved_request,
            ),
        )

    def tensorcast_instance_publish_engine_request_id(
        self: Scheduler,
        *,
        engine_request_id: str,
        ttl_ms: int | None = None,
        timeout_ms: int | None = None,
        requested_cutoff_token_count: int | None = None,
        requested_at_ms: int | None = None,
        publish_op_id: str | None = None,
    ) -> SourcePublishClosureResult:
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            raise RequestBundleStateError(
                "Tensorcast instance publish requires the Tensorcast HiCache backend"
            )
        now_ms = (
            int(requested_at_ms)
            if requested_at_ms is not None
            else int(time.time() * 1000)
        )
        normalized_engine_request_id = str(engine_request_id).strip()
        if not normalized_engine_request_id:
            raise RequestBundleStateError(
                "Tensorcast instance publish requires a non-empty engine_request_id"
            )
        resolved_publish_op_id = str(publish_op_id or "").strip() or (
            f"publish::{normalized_engine_request_id}::{now_ms}"
        )
        typed_request = manager.build_live_request_publish_request(
            logical_request_id=normalized_engine_request_id,
            publish_op_id=resolved_publish_op_id,
            requested_at_ms=now_ms,
            requested_cutoff_token_count=requested_cutoff_token_count,
            timeout_ms=timeout_ms,
            ttl_ms=60_000 if ttl_ms is None else int(ttl_ms),
        )
        return self.tensorcast_instance_publish(request=typed_request)

    def tensorcast_instance_hydrate(
        self: Scheduler,
        request: HydrateInstanceOpRequest | dict,
        publish_manifest: PublishManifestRecord | dict,
    ) -> HydratePreparedResult:
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            raise RequestBundleStateError(
                "Tensorcast instance hydrate requires the Tensorcast HiCache backend"
            )
        typed_request = HydrateInstanceOpRequest.model_validate(request)
        typed_publish_manifest = PublishManifestRecord.model_validate(publish_manifest)
        return self._require_tensorcast_instance_ops_runtime().hydrate(
            request=typed_request,
            publish_manifest=typed_publish_manifest,
            target=manager.instance_hydrate_target(
                required_ranks=typed_publish_manifest.engine_owned_manifest.payload.compatibility.required_ranks
            ),
            execute_local=lambda: manager.instance_hydrate_local(
                request=typed_request,
                publish_manifest=typed_publish_manifest,
            ),
        )

    def tensorcast_instance_evict_local(
        self: Scheduler,
        request: EvictLocalInstanceOpRequest | dict,
    ) -> EvictLocalInstanceOpResult:
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            raise RequestBundleStateError(
                "Tensorcast instance evict_local requires the Tensorcast HiCache backend"
            )
        typed_request = EvictLocalInstanceOpRequest.model_validate(request)
        return self._require_tensorcast_instance_ops_runtime().evict_local(
            request=typed_request,
            execute_local=lambda: manager.instance_evict_local(
                request=typed_request,
            ),
        )

    def _tensorcast_instance_ops_supported_for_generate(
        self: Scheduler,
        recv_req: TokenizedGenerateReqInput,
    ) -> bool:
        return (
            recv_req.mm_inputs is None
            and recv_req.input_embeds is None
            and recv_req.session_params is None
        )

    def _maybe_start_tensorcast_live_request(
        self: Scheduler,
        req: Req,
        recv_req: TokenizedGenerateReqInput,
    ) -> None:
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            return
        state = self._tensorcast_instance_state_for_req(req)
        if state is not None and state.prepared_bundle_record is not None:
            return
        if state is not None and state.live_source_request_tracked:
            return
        if not self._tensorcast_instance_ops_supported_for_generate(recv_req):
            return
        try:
            manager.start_live_request_tracking(
                logical_request_id=req.rid,
                engine_request_id=req.rid,
                prompt_token_ids=list(req.origin_input_ids),
                requested_at_ms=int(time.time() * 1000),
                batch_request_count=1,
                parallel_sampling_count=req.sampling_params.n,
                session_lineage_depth=0,
            )
        except Exception:
            logger.exception(
                "Tensorcast live source-request tracking start failed rid=%s",
                req.rid,
            )
            return
        resolved_state = self._tensorcast_instance_state_for_req(req, create=True)
        assert resolved_state is not None
        resolved_state.live_source_request_tracked = True
        self._maybe_observe_tensorcast_live_request_progress(req)

    def _maybe_observe_tensorcast_live_request_progress(
        self: Scheduler,
        req: Req,
        *,
        emitted_decode_token_count: int | None = None,
    ) -> None:
        state = self._tensorcast_instance_state_for_req(req)
        if state is None or not state.live_source_request_tracked:
            return
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            return
        try:
            manager.observe_live_request_progress(
                logical_request_id=req.rid,
                visible_prompt_token_count=min(
                    req.kv_committed_len, len(req.origin_input_ids)
                ),
                emitted_decode_token_count=(
                    len(req.output_ids)
                    if emitted_decode_token_count is None
                    else emitted_decode_token_count
                ),
                now_ms=int(time.time() * 1000),
            )
        except Exception:
            logger.exception(
                "Tensorcast live source-request progress update failed rid=%s",
                req.rid,
            )

    def _maybe_cleanup_tensorcast_live_request(self: Scheduler, req: Req) -> None:
        state = self._tensorcast_instance_state_for_req(req)
        if state is None or not state.live_source_request_tracked:
            return
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            state.live_source_request_tracked = False
            self._maybe_drop_tensorcast_instance_req_state(req, state)
            return
        try:
            manager.cleanup_live_request_tracking(
                logical_request_id=req.rid,
                now_ms=int(time.time() * 1000),
            )
        except Exception:
            logger.exception(
                "Tensorcast live source-request cleanup failed rid=%s",
                req.rid,
            )
        state.live_source_request_tracked = False
        self._maybe_drop_tensorcast_instance_req_state(req, state)

    def _maybe_bind_tensorcast_prepared_bundle(
        self: Scheduler,
        req: Req,
        recv_req: TokenizedGenerateReqInput,
    ) -> bool:
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            return True
        if recv_req.mm_inputs is not None:
            return True
        if recv_req.input_embeds is not None:
            return True
        if recv_req.session_params is not None:
            return True
        bind_result = manager.bind_prepared_bundle_for_generate(
            logical_request_id=req.rid,
            scheduler_rid=req.rid,
            prompt_token_ids=list(req.origin_input_ids),
            requested_at_ms=int(time.time() * 1000),
        )
        if bind_result.action == PreparedBundleBindAction.FAIL_CLOSED:
            req.set_finish_with_abort(
                f"Tensorcast prepared-bundle admission failed: {bind_result.reason}"
            )
            return False
        if bind_result.action != PreparedBundleBindAction.ATTACHED:
            return True
        state = self._tensorcast_instance_state_for_req(req, create=True)
        assert state is not None
        state.prepared_bundle_record = bind_result.record
        state.prepared_bundle_key = bind_result.prepared_bundle_key
        state.prepared_bundle_claim_token = bind_result.claim_token
        return True

    def _release_tensorcast_prepared_host_node(
        self: Scheduler,
        state: _SchedulerTensorcastInstanceReqState,
    ) -> None:
        if state.prepared_bundle_host_node is None:
            return
        with suppress(Exception):
            state.prepared_bundle_host_node.release_host()
        state.prepared_bundle_host_node = None

    def _install_tensorcast_prepared_prefix(
        self: Scheduler,
        *,
        req: Req,
        state: _SchedulerTensorcastInstanceReqState,
        storage_backend: TensorcastStore,
        manager: RequestBundleManager,
        record: PreparedBundleRecord,
        hold_set,
        cutoff_prompt_token_ids: tuple[int, ...],
    ) -> None:
        if not cutoff_prompt_token_ids:
            self._release_tensorcast_prepared_host_node(state)
            return
        if hold_set is None:
            raise RequestBundleStateError(
                "prepared bundle is missing host hold refs for the requested cutoff"
            )
        tree_cache = self.tree_cache
        insert_host_prefix = getattr(tree_cache, "insert_host_prefix", None)
        if insert_host_prefix is None:
            raise RequestBundleStateError(
                "ordinary instance consume requires a host-aware HiRadix cache"
            )
        ordered_refs = tuple(
            sorted(hold_set.refs, key=lambda hold_ref: hold_ref.logical_page_index)
        )
        expected_page_indices = tuple(range(len(ordered_refs)))
        actual_page_indices = tuple(
            hold_ref.logical_page_index for hold_ref in ordered_refs
        )
        if actual_page_indices != expected_page_indices:
            raise RequestBundleStateError(
                "prepared hold refs must be contiguous and start at logical_page_index=0"
            )
        key = RadixKey(
            token_ids=list(cutoff_prompt_token_ids),
            extra_key=req.extra_key,
        )
        page_hashes = [hold_ref.page_hash for hold_ref in ordered_refs]
        host_indices = manager.host_indices_for_prepared_hold_refs(ordered_refs)
        matched_length = int(insert_host_prefix(key, host_indices, page_hashes))
        if matched_length % storage_backend.page_size != 0:
            raise RequestBundleStateError(
                "prepared host-prefix insertion returned a non-page-aligned match length"
            )
        matched_page_count = matched_length // storage_backend.page_size
        if matched_page_count > 0:
            manager.release_prepared_hold_refs(ordered_refs[:matched_page_count])
        match_result = tree_cache.match_prefix(MatchPrefixParams(key=key))
        matched_prefix_tokens = len(match_result.device_indices) + int(
            match_result.host_hit_length
        )
        if matched_prefix_tokens != len(cutoff_prompt_token_ids):
            raise RequestBundleStateError(
                "prepared bundle did not become fully visible through the ordinary prefix path"
            )
        self._release_tensorcast_prepared_host_node(state)
        if int(match_result.host_hit_length) > 0:
            match_result.last_host_node.protect_host()
            state.prepared_bundle_host_node = match_result.last_host_node

    def _maybe_consume_tensorcast_prepared_bundle(self: Scheduler, req: Req) -> None:
        state = self._tensorcast_instance_state_for_req(req)
        if state is None or state.prepared_bundle_record is None:
            return
        if state.prepared_bundle_record.state == PreparedBundleLifecycleState.CONSUMED:
            return
        storage_backend = self._tensorcast_storage_backend()
        manager = self._tensorcast_request_bundle_manager()
        if storage_backend is None or manager is None:
            return
        try:
            consumed_record = manager.consume_attached_prepared_bundle_for_generate(
                logical_request_id=req.rid,
                publish_manifest_digest=state.prepared_bundle_record.publish_manifest_digest,
                prompt_token_ids=list(req.origin_input_ids),
                requested_at_ms=int(time.time() * 1000),
                install_prepared_bundle=lambda record,
                hold_set,
                cutoff_prompt_token_ids: self._install_tensorcast_prepared_prefix(
                    req=req,
                    state=state,
                    storage_backend=storage_backend,
                    manager=manager,
                    record=record,
                    hold_set=hold_set,
                    cutoff_prompt_token_ids=cutoff_prompt_token_ids,
                ),
            )
            state.prepared_bundle_record = consumed_record
        except Exception as exc:
            logger.exception(
                "Tensorcast prepared-bundle consume failed rid=%s manifest=%s",
                req.rid,
                state.prepared_bundle_record.publish_manifest_digest,
            )
            self._release_tensorcast_prepared_host_node(state)
            with suppress(Exception):
                manager.cleanup_bound_prepared_bundle(
                    publish_manifest_digest=state.prepared_bundle_record.publish_manifest_digest,
                    requested_at_ms=int(time.time() * 1000),
                )
            state.prepared_bundle_record = None
            state.prepared_bundle_key = None
            state.prepared_bundle_claim_token = None
            self._maybe_drop_tensorcast_instance_req_state(req, state)
            req.set_finish_with_abort(
                f"Tensorcast prepared-bundle consume failed: {exc}"
            )

    def _maybe_cleanup_tensorcast_prepared_bundle(self: Scheduler, req: Req) -> None:
        state = self._tensorcast_instance_state_for_req(req)
        if state is None or state.prepared_bundle_record is None:
            return
        prepared_bundle_record = state.prepared_bundle_record
        self._release_tensorcast_prepared_host_node(state)
        manager = self._tensorcast_request_bundle_manager()
        if manager is None:
            state.prepared_bundle_record = None
            state.prepared_bundle_key = None
            state.prepared_bundle_claim_token = None
            self._maybe_drop_tensorcast_instance_req_state(req, state)
            return
        manager.cleanup_bound_prepared_bundle(
            publish_manifest_digest=prepared_bundle_record.publish_manifest_digest,
            requested_at_ms=int(time.time() * 1000),
        )
        state.prepared_bundle_record = None
        state.prepared_bundle_key = None
        state.prepared_bundle_claim_token = None
        self._maybe_drop_tensorcast_instance_req_state(req, state)
