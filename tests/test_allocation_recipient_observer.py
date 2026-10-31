"""Ordinary CPU allocation fragments and passive recorder controls.

Owner: Hari Krishna Kancharla.
No serving process, device, transfer, failure schedule or ownership stimulus runs.
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import shutil
from types import CodeType, FunctionType, SimpleNamespace

import pytest

from sdcguard.lifetime import allocation_recipient_obligations as R
from sdcguard.lifetime import assignment_obligations as A
from sdcguard.lifetime import allocation_recipient_recording as P
from sdcguard.lifetime import mapping_closure as C
from sdcguard.lifetime import vllm_forward as V
from sdcguard.lifetime import vllm_nixl as L
from sdcguard.lifetime.allocation_recipient_observer import AllocationRecipientObserver, _fingerprint
from test_assignment_init import Fields, fragment, primary
from test_mapping_observer import observer_environment, source_class


@dataclass
class KVCacheBlocks:
    blocks: tuple


class RequestStatus(Enum):
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3


def recipient_environment(primary, tmp_path, *, lifetime=False):
    _, sources = primary
    coord = source_class(sources['vllm/v1/core/kv_cache_coordinator.py'], 'KVCacheCoordinator',
        {'allocate_new_computed_blocks', 'allocate_new_blocks', 'cache_blocks', 'get_blocks'},
        dict(CrossAttentionManager=type('CrossAttentionManager', (), {}), replay_boundaries=lambda *args, **kwargs: None))
    coord.__module__ = 'vllm.v1.core.kv_cache_coordinator'
    def get_count(self, **args):
        group = self.single_type_managers[0]
        return max(0, (args['num_tokens'] + 15) // 16 - len(group.req_to_blocks.get(args['request_id'], [])))
    coord.get_num_blocks_to_allocate = get_count
    coord.remove_skipped_blocks = lambda *args, **kwargs: None
    def setup(manager):
        manager.max_model_len = 64
        manager.watermark_blocks = 0
        manager.retained_hit_group_ids = ()
        manager.empty_kv_cache_blocks = KVCacheBlocks(((),))
        manager.coordinator.scheduler_block_size = 16
        manager.coordinator.eagle_group_ids = set()
    def build(value, runner_class, forward_dir, engine, profile_types, context, factory, directory, source_root):
        # Recipient coverage requires complete primary originals, even for
        # paths this ordinary allocation fixture never invokes.
        primary_scheduler = source_class(sources['vllm/v1/core/sched/scheduler.py'], 'Scheduler',
                                         {'schedule'}, {})
        module_code = compile(sources['vllm/v1/core/sched/scheduler.py'], 'vllm/v1/core/sched/scheduler.py',
                              'exec', dont_inherit=True)
        scheduler_code = next(code for code in module_code.co_consts
                              if type(code) is CodeType and code.co_name == 'Scheduler')
        schedule_code = next(code for code in scheduler_code.co_consts
                             if type(code) is CodeType and code.co_name == 'schedule')
        assert schedule_code.co_freevars == ()
        schedule = FunctionType(schedule_code, primary_scheduler.schedule.__globals__, 'schedule',
                                primary_scheduler.schedule.__defaults__)
        schedule.__kwdefaults__ = primary_scheduler.schedule.__kwdefaults__
        schedule.__annotations__ = primary_scheduler.schedule.__annotations__
        value.bindings.scheduler.schedule = schedule
        native = A.adapter_contract('vllm.0.32.uniproc.eager.native.v1')
        pool_methods = {label.split('.')[1] for label in
                        native['mutation_methods'] | native['guard_methods']
                        if label.startswith('BlockPool.')}
        primary_pool = source_class(sources['vllm/v1/core/block_pool.py'], 'BlockPool', pool_methods,
            dict(BlockHashToBlockMap=dict, logger=Fields(warning=lambda *a: None, info=lambda *a: None)))
        for name in pool_methods:
            setattr(value.bindings.pool, name, getattr(primary_pool, name))
        if value.bindings.coordinator is not coord and 'get_blocks' in vars(value.bindings.coordinator):
            delattr(value.bindings.coordinator, 'get_blocks')
        value.bindings.manager_module.KVCacheBlocks = KVCacheBlocks
        namespace = dict(RequestStatus=RequestStatus, KVCacheBlocks=KVCacheBlocks)
        for name in ('allocate_slots', 'create_kv_cache_blocks'):
            setattr(value.bindings.manager, name, fragment(sources['vllm/v1/core/kv_cache_manager.py'], name,
                                                          dict(namespace), 'KVCacheManager'))
        base = value.bindings.group_manager.__mro__[1]
        base.get_num_skipped_tokens = fragment(sources['vllm/v1/core/single_type_kv_cache_manager.py'],
                                              'get_num_skipped_tokens', {}, 'SingleTypeKVCacheManager')
        value.config.scheduler_config.async_scheduling = False
        value.config.scheduler_config.scheduler_cls = None
        if lifetime:
            L.install_manager(value.bindings.manager)
        return V.Installation(runner_class, forward_dir, engine, profile_types, context,
            assignment_factory=lambda registration, health: factory(registration, health, closure_callback=C.write_receipt)).attach()
    value = observer_environment(primary, tmp_path, journal=True, recipient_factory=AllocationRecipientObserver,
        runtime_options=dict(coordinator_base=coord, manager_setup=setup), installation_builder=build)
    value.scheduler.defer_block_free = False
    for request in value.scheduler.requests.values():
        request.sampling_params = Fields(n=1)
        request.pooling_params = request.lora_request = None
        request.num_preemptions = request.replay_start = request.shared_prefix_boundary = request.num_in_flight_tokens = 0
        request.num_computed_tokens = 0
        request.num_prompt_tokens = 16
        request._all_token_ids = [0] * 16
        request.status = RequestStatus.WAITING
    return value


def allocate(value, *, external=False):
    request = next(iter(value.scheduler.requests.values()))
    manager = value.scheduler.kv_cache_manager
    before = len(manager.block_pool.free_block_queue.blocks)
    kwargs = dict(num_external_computed_tokens=16, delay_cache_blocks=True, skip_zeroing_group_ids=(0,)) if external else {}
    result = manager.allocate_slots(request, 0 if external else 16, **kwargs)
    assert type(result) is KVCacheBlocks
    assert len(manager.block_pool.free_block_queue.blocks) == before - 1
    assert manager.coordinator.single_type_managers[0].req_to_blocks[request.request_id][0] is manager.block_pool.blocks[1]
    assert result is manager.empty_kv_cache_blocks if external else result.blocks[0][0] is manager.block_pool.blocks[1]
    return result


@pytest.mark.parametrize('external', [False, True])
@pytest.mark.parametrize('lifetime', [False, True])
def test_real_ordinary_source_returns_produce_only_capacity_recipient_facts(primary, tmp_path, external, lifetime):
    value = recipient_environment(primary, tmp_path, lifetime=lifetime)
    try:
        installed = value.bindings.manager.allocate_slots
        if lifetime:
            assert 'KVCacheManager.allocate_slots' in L._installed_hook_names()
        allocate(value, external=external)
        observer = value.mapping.recipient
        assert not observer.status['errors']
        assert value.bindings.manager.allocate_slots is installed
        value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        frozen = P.prepare(observer, paired['payload'], paired['receipts'])
        report = frozen['report']
        assert report['status'] == 'observed', report['missing']
        assert len(report['receipts']) == 1
        assert report['receipts'][0]['method'].endswith('allocate_external_computed_blocks' if external else 'allocate_new_blocks')
        assert report['receipts'][0]['request_incarnation'] == next(r for r in observer.records if r['kind']=='allocation_begin')['request_incarnation']
        assert report['scope_verdict'] == 'inconclusive'
        assert report['read_rights_verdict'] == report['lease_join_verdict'] == report['physical_read_verdict'] == 'unknown'
        if external:
            assert next(r for r in observer.records if r['kind']=='allocation_end')['returned'] == [[]]
        if lifetime:
            assert 'KVCacheManager.allocate_slots' in L._installed_hook_names()
    finally:
        value.installation.detach()
        L.reset()


def test_actual_capture_export_copy_and_reverification_use_paired_settled_bytes(primary, tmp_path):
    value = recipient_environment(primary, tmp_path)
    try:
        allocate(value)
        value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        exported = P.write_capture(value.mapping.recipient, tmp_path/'portable', paired['payload'], paired['receipts'])
        assert exported['report']['status'] == 'observed'
        shutil.copytree(tmp_path/'portable', tmp_path/'moved')
        assert P.verify(tmp_path/'moved') == {k:v for k,v in exported.items() if k!='path'}
        raw = tmp_path/'moved'/'recipient.jsonl'
        raw.write_bytes(raw.read_bytes() + b' ')
        with pytest.raises(ValueError):
            P.verify(tmp_path/'moved')
    finally:
        value.installation.detach()


def test_missing_settled_receipt_is_unknown_and_partial_export_is_preserved(primary, tmp_path):
    value = recipient_environment(primary, tmp_path)
    try:
        allocate(value)
        value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        report = P.prepare(value.mapping.recipient, paired['payload'], {})['report']
        assert report['status'] == 'unknown' and not report['receipts']
        output = P.write_capture(value.mapping.recipient, tmp_path/'partial', paired['payload'], {})
        assert output['report']['status'] == 'unknown'
        assert (tmp_path/'partial'/'recipient.jsonl.incomplete').is_file()
        assert P.verify(tmp_path/'partial')['report']['read_rights_verdict'] == 'unknown'
    finally:
        value.installation.detach()


def test_failed_export_consumes_attempt_and_preserves_existing_destination(primary, tmp_path):
    value = recipient_environment(primary, tmp_path)
    try:
        allocate(value);value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        existing = tmp_path/'existing';existing.mkdir()
        with pytest.raises(FileExistsError):
            P.write_capture(value.mapping.recipient, existing, paired['payload'], paired['receipts'])
        assert value.mapping.recipient.status['export_attempted'] is True
        assert value.mapping.recipient.status['exported'] is False
        assert value.mapping.recipient.status['export_in_progress'] is False
        with pytest.raises(ValueError, match='already reserved'):
            P.write_capture(value.mapping.recipient, tmp_path/'clean', paired['payload'], paired['receipts'])
        assert not (tmp_path/'clean').exists()
        assert not list(existing.iterdir())
    finally:
        value.installation.detach()


def test_recorder_error_does_not_change_normal_native_return_or_arguments(primary, tmp_path, monkeypatch):
    value = recipient_environment(primary, tmp_path)
    try:
        recipient = value.mapping.recipient
        def broken(*args, **kwargs):
            raise OSError('ordinary recorder failure')
        monkeypatch.setattr(recipient, '_emit', broken)
        result = allocate(value)
        assert type(result) is KVCacheBlocks
        assert recipient.status['errors'] > 0
        value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        assert P.prepare(recipient, paired['payload'], paired['receipts'])['report']['status'] == 'unknown'
    finally:
        value.installation.detach()


def test_explicit_composition_requires_exact_tracked_slots_and_detaches(primary, tmp_path):
    value = recipient_environment(primary, tmp_path, lifetime=True)
    try:
        recipient = value.mapping.recipient
        assert recipient.coverage()
        owner, name, inner, outer = recipient._compositions[0]
        assert L._composition_matches(owner, name, inner, outer)
        with pytest.raises(ValueError):
            L.register_composition(owner, name, inner, lambda *args: None)
        allocate(value);value.runner.shutdown()
        assert getattr(owner, name) is inner
        assert not L._COMPOSED_HOOKS.get(owner)
        assert 'KVCacheManager.allocate_slots' in L._installed_hook_names()
    finally:
        value.installation.detach();L.reset()


def test_admission_rejection_counts_real_dropped_records_without_changing_allocation(primary, tmp_path, monkeypatch):
    value = recipient_environment(primary, tmp_path)
    try:
        with monkeypatch.context() as change:
            change.setattr(R, 'MAX_RECORD_BYTES', 1)
            allocate(value)
        recipient = value.mapping.recipient
        assert recipient.status['dropped'] > 0 and recipient.status['errors'] > 0
        value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        assert P.prepare(recipient, paired['payload'], paired['receipts'])['report']['status'] == 'unknown'
    finally:
        value.installation.detach()


def test_registered_composition_loss_is_sticky_for_exact_slot_and_unrelated_labels_remain(primary, tmp_path):
    value = recipient_environment(primary, tmp_path, lifetime=True)
    class UnrelatedManager:
        def allocate_slots(self, *args, **kwargs):
            return None
    def unchanged(function):
        def delegated(*args, **kwargs):
            return function(*args, **kwargs)
        return delegated
    assert L._wrap(UnrelatedManager, 'allocate_slots', unchanged, 'KVCacheManager.allocate_slots')
    try:
        recipient = value.mapping.recipient
        allocate(value)
        owner, name, inner, outer = recipient._compositions[0]
        original = outer.__code__
        try:
            # This replacement is never executed. Only coverage bookkeeping is tested.
            outer.__code__ = original.replace(co_filename=original.co_filename + '.changed')
            assert not L._composition_matches(owner, name, inner, outer)
            assert getattr(owner, name) is outer
            assert L._INSTALLED_HOOKS[UnrelatedManager][name][0] is getattr(UnrelatedManager, name)
            assert 'KVCacheManager.allocate_slots' in L._installed_hook_names()
            assert not recipient.coverage()
        finally:
            outer.__code__ = original
        assert not L._composition_matches(owner, name, inner, outer)
        assert not recipient.coverage()
        value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        report = P.prepare(recipient, paired['payload'], paired['receipts'])['report']
        assert report['status'] == 'unknown' and not report['receipts']
        assert recipient._coverage_stable is False
    finally:
        value.installation.detach();L.reset()


def test_hash_consistent_missing_records_remain_unknown_and_boolean_size_is_refused(primary, tmp_path):
    value = recipient_environment(primary, tmp_path)
    try:
        allocate(value);value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        P.write_capture(value.mapping.recipient, tmp_path/'source-packet', paired['payload'], paired['receipts'])
        shutil.copytree(tmp_path/'source-packet', tmp_path/'missing-records')
        root = tmp_path/'missing-records'
        manifest = json.loads((root/P.MANIFEST).read_bytes())
        raw_path = root/manifest['recipient_file']
        raw_path.write_bytes(b'')
        manifest['files'][manifest['recipient_file']] = P._identity(b'')
        (root/P.MANIFEST).write_bytes(P.io.json_bytes(manifest))
        assert P.verify(root)['report']['status'] == 'unknown'
        manifest['files'][manifest['recipient_file']]['bytes'] = False
        (root/P.MANIFEST).write_bytes(P.io.json_bytes(manifest))
        with pytest.raises(ValueError, match='malformed'):
            P.verify(root)
    finally:
        value.installation.detach()


def test_invalid_recipient_factory_refuses_before_assignment_without_leaking_hooks(primary, tmp_path):
    value = observer_environment(primary, tmp_path, recipient_factory=lambda mapping: object())
    try:
        assert value.installation.bridge is None
        assert value.installation.bridge_error == 'assignment setup: MappingGap'
        for cls, name in ((value.bindings.pool, 'get_new_blocks'),
                          (value.bindings_mapping.manager_base, 'allocate_new_blocks')):
            assert getattr(getattr(cls, name), '_mapping_observer', None) is None
    finally:
        value.installation.detach()
