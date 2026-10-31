"""Exact passive wrapper-chain bookkeeping. Owner: Hari Krishna Kancharla.

Detached controls never invoke a changed callback. The ordinary source control
allocates before the metadata-only change and restores code before shutdown.
No engine, request experiment, device operation or failure body is executed.
"""
from types import SimpleNamespace

import pytest

from sdcguard.lifetime import allocation_recipient_recording as recording
from sdcguard.lifetime import mapping_closure
from sdcguard.lifetime import vllm_nixl as lifetime
from sdcguard.lifetime.allocation_recipient_observer import (
    AllocationRecipientObserver, OUTER, COMPUTED, COORDINATED, LOCAL, EXTERNAL, ORDINARY, _fingerprint)
from sdcguard.lifetime.mapping_observer import MappingObserver
from test_allocation_recipient_observer import recipient_environment, allocate
from test_assignment_init import primary

POOL = 'BlockPool.get_new_blocks'
SCHEDULE = 'Scheduler.schedule'
HOOKS = (OUTER, COMPUTED, COORDINATED, LOCAL, EXTERNAL, ORDINARY, POOL, SCHEDULE)


def detached(composed=False):
    class Manager:
        def allocate_slots(self, request, *args, **kwargs):
            return request
        def create_kv_cache_blocks(self, blocks):
            return blocks
    class Coordinator:
        def allocate_new_computed_blocks(self, request, *args, **kwargs):
            return None
        def allocate_new_blocks(self, request, *args, **kwargs):
            return None
        def get_blocks(self, request):
            return ()
    class Group:
        def add_local_computed_blocks(self, request, *args, **kwargs):
            return None
        def allocate_external_computed_blocks(self, request, *args, **kwargs):
            return None
        def allocate_new_blocks(self, request, *args, **kwargs):
            return []
        def get_num_skipped_tokens(self, request):
            return 0
    class Pool:
        def get_new_blocks(self, count):
            return []
    class Scheduler:
        def schedule(self):
            return None
    mapping = object.__new__(MappingObserver)
    mapping.registry = SimpleNamespace(new_token=lambda: '1' * 32)
    mapping._hooks = []
    mapping._getter_owner, mapping._getter = Coordinator, Coordinator.get_blocks
    mapping.methods = {label: 'unused by detached coverage' for label in (LOCAL, EXTERNAL, ORDINARY, POOL)}
    observer = AllocationRecipientObserver(mapping)
    mapping.recipient = observer
    owners = {OUTER: Manager, COMPUTED: Coordinator, COORDINATED: Coordinator,
              LOCAL: Group, EXTERNAL: Group, ORDINARY: Group, POOL: Pool, SCHEDULE: Scheduler}
    natives = {label: getattr(owner, label.split('.')[1]) for label, owner in owners.items()}
    if composed:
        lifetime.install_manager(Manager)
    for label, owner in owners.items():
        name = label.split('.')[1]
        original = getattr(owner, name)
        if label in (OUTER, COMPUTED, COORDINATED):
            make = lambda function, label=label: observer.wrap(label, function)
        elif label == SCHEDULE:
            make = mapping._schedule_wrapper
        else:
            make = lambda function, label=label: mapping._mutation_wrapper(label, function)
        mapping._install(owner, name, make)
        installed = getattr(owner, name)
        if label != SCHEDULE:
            observer._slots[label] = (owner, name, installed)
            observer._codes[label] = (natives[label], _fingerprint(natives[label].__code__))
        if composed and label == OUTER:
            lifetime.register_composition(owner, name, original, installed)
            observer._compositions.append((owner, name, original, installed))
    for label, owner, name in (
            ('KVCacheManager.create_kv_cache_blocks', Manager, 'create_kv_cache_blocks'),
            ('SingleTypeKVCacheManager.get_num_skipped_tokens', Group, 'get_num_skipped_tokens')):
        function = getattr(owner, name)
        observer._slots[label] = (owner, name, function)
        observer._codes[label] = (function, _fingerprint(function.__code__))
    return SimpleNamespace(mapping=mapping, observer=observer, owners=owners, natives=natives)


def cleanup(value):
    value.mapping._cleanup_hooks()
    lifetime.reset()


def wrapped(value, label, layer):
    owner, name = value.owners[label], label.split('.')[1]
    installed = getattr(owner, name)
    return installed if layer == 'installed' else lifetime._closure(installed)['supplied']


def outer_still_valid(value, composed, label):
    if composed and label != OUTER:
        assert lifetime._composition_matches(*value.observer._compositions[0])


@pytest.mark.parametrize('composed', [False, True])
@pytest.mark.parametrize('label', HOOKS)
@pytest.mark.parametrize('layer', ['installed', 'supplied'])
def test_every_installed_and_supplied_code_identity_is_checked_fresh(composed, label, layer):
    value = detached(composed)
    try:
        observer = value.observer
        assert observer.coverage()
        function = wrapped(value, label, layer)
        code = function.__code__
        try:
            function.__code__ = code.replace(co_filename=code.co_filename + '.changed')
            outer_still_valid(value, composed, label)
            assert observer.coverage() is False
        finally:
            function.__code__ = code
        assert observer.coverage() is False
        assert observer._coverage_stable is False and observer.status['healthy'] is False
    finally:
        cleanup(value)


_CELL_CASES = [(label, layer, name) for label in HOOKS
               for layer, names in (
                   ('installed', ('original', 'self', 'supplied')),
                   ('supplied', ('original', 'self') if label == SCHEDULE else ('original', 'self', 'signature', 'label')))
               for name in names]


@pytest.mark.parametrize('composed', [False, True])
@pytest.mark.parametrize('label,layer,name', _CELL_CASES)
def test_every_closed_delegation_cell_is_checked_fresh(composed, label, layer, name):
    value = detached(composed)
    try:
        observer = value.observer
        assert observer.coverage()
        function = wrapped(value, label, layer)
        cell = dict(zip(function.__code__.co_freevars, function.__closure__))[name]
        original = cell.cell_contents
        try:
            # Neither the changed callback nor the redirected ordinary delegate runs.
            replacement = value.natives[OUTER] if name in ('original', 'supplied') else object()
            if replacement is original:
                replacement = value.natives[COMPUTED]
            assert replacement is not original
            cell.cell_contents = replacement
            outer_still_valid(value, composed, label)
            assert observer.coverage() is False
        finally:
            cell.cell_contents = original
        assert observer.coverage() is False and observer._coverage_stable is False
    finally:
        cleanup(value)


@pytest.mark.parametrize('kind', ['installed_code', 'supplied_code', 'original_cell', 'self_cell', 'signature_cell', 'label_cell'])
def test_first_helper_capture_refuses_an_already_unsupported_template_or_delegate(kind):
    value = detached()
    observer = value.observer
    function = wrapped(value, COORDINATED, 'installed' if kind == 'installed_code' else 'supplied')
    original = function.__code__ if kind.endswith('_code') else None
    cell = None
    try:
        if original is not None:
            function.__code__ = original.replace(co_filename=original.co_filename + '.changed')
        else:
            name = kind.removesuffix('_cell')
            cell = dict(zip(function.__code__.co_freevars, function.__closure__))[name]
            original = cell.cell_contents
            cell.cell_contents = object()
        assert observer._chains is None
        assert observer.coverage() is False and observer._coverage_stable is False
        assert observer._chains is None
    finally:
        if cell is None:
            function.__code__ = original
        else:
            cell.cell_contents = original
    assert observer.coverage() is False and observer._chains is None
    cleanup(value)


@pytest.mark.parametrize('composed', [False, True])
@pytest.mark.parametrize('label', ['KVCacheManager.create_kv_cache_blocks', 'SingleTypeKVCacheManager.get_num_skipped_tokens', 'getter'])
def test_direct_helper_and_getter_code_objects_are_frozen(composed, label):
    value = detached(composed)
    try:
        observer = value.observer
        assert observer.coverage()
        function = value.mapping._getter if label == 'getter' else observer._slots[label][2]
        code = function.__code__
        try:
            function.__code__ = code.replace(co_filename=code.co_filename + '.changed')
            assert observer.coverage() is False
        finally:
            function.__code__ = code
        assert observer.coverage() is False
    finally:
        cleanup(value)


@pytest.mark.parametrize('kind', ['hook_inventory', 'dynamic_recipient', 'getter_pointer', 'getter_owner', 'slot_map', 'code_map', 'composition_inventory'])
def test_inventory_and_owner_references_cannot_be_rebound_after_capture(kind):
    value = detached(True)
    try:
        observer = value.observer
        assert observer.coverage()
        if kind == 'hook_inventory':
            value.mapping._hooks.append(value.mapping._hooks[-1])
            restore = lambda: value.mapping._hooks.pop()
        elif kind == 'dynamic_recipient':
            value.mapping.recipient = None
            restore = lambda: setattr(value.mapping, 'recipient', observer)
        elif kind in ('getter_pointer', 'getter_owner'):
            name = '_getter' if kind == 'getter_pointer' else '_getter_owner'
            old = getattr(value.mapping, name)
            setattr(value.mapping, name, object())
            restore = lambda: setattr(value.mapping, name, old)
        elif kind == 'composition_inventory':
            entry = observer._compositions.pop()
            restore = lambda: observer._compositions.append(entry)
        else:
            values = observer._slots if kind == 'slot_map' else observer._codes
            entry = values.pop(COMPUTED)
            restore = lambda: values.__setitem__(COMPUTED, entry)
        try:
            assert observer.coverage() is False
        finally:
            restore()
        assert observer.coverage() is False
    finally:
        cleanup(value)


def test_first_capture_refuses_duplicate_hook_slots():
    value = detached()
    value.mapping._hooks.append(value.mapping._hooks[-1])
    try:
        assert value.observer.coverage() is False
        value.mapping._hooks.pop()
        assert value.observer.coverage() is False and value.observer._chains is None
    finally:
        cleanup(value)


@pytest.mark.parametrize('composed', [False, True])
def test_observed_lower_chain_loss_keeps_actual_ordinary_capture_unknown_after_restore(primary, tmp_path, composed):
    value = recipient_environment(primary, tmp_path, lifetime=composed)
    try:
        recipient = value.mapping.recipient
        assert recipient._chains is not None
        allocate(value)
        owner = value.bindings.pool
        function = lifetime._closure(owner.get_new_blocks)['supplied']
        code = function.__code__
        try:
            function.__code__ = code.replace(co_filename=code.co_filename + '.changed')
            assert recipient.coverage() is False
        finally:
            function.__code__ = code
        assert recipient.coverage() is False
        value.runner.shutdown()
        paired = mapping_closure.capture_closed(value.journal_directory)
        result = recording.prepare(recipient, paired['payload'], paired['receipts'])['report']
        assert result['metadata_completeness'] == 'unknown' and not result['receipts']
        assert result['read_rights_verdict'] == result['lease_join_verdict'] == result['physical_read_verdict'] == 'unknown'
    finally:
        value.installation.detach()
        lifetime.reset()
