"""Pinned native originals before attachment. Owner: Hari Krishna Kancharla.

Complete primary modules are compiled only into code objects. Fresh local
classes receive FunctionType objects with actual defining-class cells. No
primary function body or changed callback is invoked by these controls.
"""
import hashlib
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace
import zipfile

import pytest

from sdcguard.lifetime import allocation_recipient_obligations as contract
from sdcguard.lifetime import assignment_obligations as assignment
from sdcguard.lifetime.allocation_recipient_observer import AllocationRecipientObserver, _fingerprint
from sdcguard.lifetime.mapping_observer import MappingObserver


_PATHS = {
    'KVCacheManager': 'vllm/v1/core/kv_cache_manager.py',
    'KVCacheCoordinator': 'vllm/v1/core/kv_cache_coordinator.py',
    'SingleTypeKVCacheManager': 'vllm/v1/core/single_type_kv_cache_manager.py',
    'FullAttentionManager': 'vllm/v1/core/single_type_kv_cache_manager.py',
    'BlockPool': 'vllm/v1/core/block_pool.py',
    'Scheduler': 'vllm/v1/core/sched/scheduler.py',
}
_DESCRIPTION = assignment.adapter_contract('vllm.0.32.uniproc.eager.native.v1')
_LABELS = tuple(sorted(set(_DESCRIPTION['mutation_methods']) | set(_DESCRIPTION['guard_methods']) | {
    'KVCacheManager.allocate_slots', 'KVCacheManager.create_kv_cache_blocks',
    'KVCacheCoordinator.allocate_new_computed_blocks', 'KVCacheCoordinator.allocate_new_blocks',
    'KVCacheCoordinator.get_blocks', 'SingleTypeKVCacheManager.get_num_skipped_tokens', 'Scheduler.schedule'}))


def cell(value):
    return (lambda: value).__closure__[0]


@pytest.fixture(scope='module')
def primary_objects():
    source = Path(__file__).parent
    with zipfile.ZipFile(source / 'fixtures/assignment_init/primary.zip') as archive:
        raw = {name: archive.read(name) for name in archive.namelist() if not name.endswith('/')}
    raw['vllm/v1/request.py'] = (source / 'fixtures/assignment_native/request.txt').read_bytes()
    pins = contract.source_pins()
    assert len(pins) == 36 and set(raw) == set(pins)
    assert all(hashlib.sha256(raw[name]).hexdigest() == digest for name, digest in pins.items())
    # compile creates code objects; no module is executed or imported.
    modules = {path: compile(raw[path], path, 'exec', dont_inherit=True) for path in set(_PATHS.values())}
    codes = {}
    for label in _LABELS:
        owner, name = label.split('.')
        classes = [value for value in modules[_PATHS[owner]].co_consts
                   if type(value) is CodeType and value.co_name == owner]
        assert len(classes) == 1
        functions = [value for value in classes[0].co_consts
                     if type(value) is CodeType and value.co_name == name]
        assert len(functions) == 1 and all(name == '__class__' for name in functions[0].co_freevars)
        codes[label] = functions[0]
    assert len(codes) == 22
    return SimpleNamespace(raw=raw, codes=codes, pins=pins)


def environment(primary_objects, tmp_path, alter=None):
    for name, raw in primary_objects.raw.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    classes = {}
    for name in _PATHS:
        bases = (classes['SingleTypeKVCacheManager'],) if name == 'FullAttentionManager' else ()
        classes[name] = type(name, bases, {})
    classes['KVCacheCoordinatorNoPrefixCache'] = type(
        'KVCacheCoordinatorNoPrefixCache', (classes['KVCacheCoordinator'],), {})
    natives = {}
    for label, code in primary_objects.codes.items():
        owner, name = label.split('.')
        closed = tuple(cell(classes[owner]) for _ in code.co_freevars)
        function = FunctionType(code, {'__name__': 'local_inert_primary'}, name, closure=closed or None)
        setattr(classes[owner], name, function)
        natives[label] = function
    if alter is not None:
        alter(classes, natives)
    mapping = object.__new__(MappingObserver)
    mapping.registry = SimpleNamespace(new_token=lambda: '1' * 32)
    mapping._hooks = []
    mapping.methods = {**_DESCRIPTION['mutation_methods'], **_DESCRIPTION['guard_methods']}
    mapping.coverage_names = sorted(mapping.methods)
    mapping.source_root = tmp_path
    mapping.bindings = SimpleNamespace(initialization=SimpleNamespace(
        manager=classes['KVCacheManager'], coordinator=classes['KVCacheCoordinatorNoPrefixCache'],
        pool=classes['BlockPool'], group_manager=classes['FullAttentionManager'], scheduler=classes['Scheduler']),
        manager_base=classes['SingleTypeKVCacheManager'])
    mapping._getter_owner = classes['KVCacheCoordinator']
    mapping._getter = natives['KVCacheCoordinator.get_blocks']
    mapping._attach()
    observer = AllocationRecipientObserver(mapping)
    mapping.recipient = observer
    return SimpleNamespace(mapping=mapping, observer=observer, natives=natives, classes=classes)


def assert_originals_restored(value):
    assert value.mapping._hooks == []
    for label, function in value.natives.items():
        owner, name = label.split('.')
        assert vars(value.classes[owner])[name] is function
        assert not hasattr(function, '_mapping_observer')
    assert value.observer._compositions == []


@pytest.mark.parametrize('label', _LABELS)
def test_each_native_original_mismatch_refuses_before_first_attachment_and_keeps_that_original(
        primary_objects, tmp_path, label):
    value = environment(primary_objects, tmp_path)
    function = value.natives[label]
    original = function.__code__
    changed = original.replace(co_consts=original.co_consts + ('inert unused metadata marker',))
    assert _fingerprint(changed) != _fingerprint(original)
    function.__code__ = changed
    try:
        with pytest.raises(ValueError, match='native original differs'):
            value.observer.attach()
        assert_originals_restored(value)
        assert value.natives[label] is function and function.__code__ is changed
        assert value.observer._native_sources is None and value.observer._chains is None
        assert value.observer.records == []
    finally:
        function.__code__ = original
        value.mapping._cleanup_hooks()


def test_all_twenty_two_primary_originals_attach_with_defining_owner_and_class_cell(primary_objects, tmp_path):
    value = environment(primary_objects, tmp_path)
    function = value.natives['FullAttentionManager.cache_blocks']
    assert function.__code__.co_freevars == ('__class__',)
    assert function.__closure__[0].cell_contents is value.classes['FullAttentionManager']
    try:
        value.observer.attach()
        assert value.observer.coverage() is True
        assert set(value.observer._native_sources) == set(value.observer._codes) == set(value.observer._slots) == set(_LABELS)
        assert value.observer.records == []  # Metadata attachment is not an engine recording.
    finally:
        value.mapping._cleanup_hooks()
    assert_originals_restored(value)


def test_coordinator_source_methods_use_legitimate_inherited_base_definition(primary_objects, tmp_path):
    value = environment(primary_objects, tmp_path)
    subclass = value.classes['KVCacheCoordinatorNoPrefixCache']
    base = value.classes['KVCacheCoordinator']
    names = ('allocate_new_computed_blocks', 'allocate_new_blocks', 'get_blocks')
    assert subclass.__mro__[1] is base and all(name not in vars(subclass) for name in names)
    try:
        value.observer.attach()
        for name in names:
            source = value.observer._native_sources['KVCacheCoordinator.' + name]
            assert source[0] is base and source[2] is value.natives['KVCacheCoordinator.' + name]
            assert name not in vars(subclass)
        assert value.observer.coverage()
    finally:
        value.mapping._cleanup_hooks()
    assert_originals_restored(value)


@pytest.mark.parametrize('wrong_owner', ['SingleTypeKVCacheManager', 'KVCacheManager'])
def test_wrong_defining_class_cell_refuses_without_changing_native_function(primary_objects, tmp_path, wrong_owner):
    value = environment(primary_objects, tmp_path)
    function = value.natives['FullAttentionManager.cache_blocks']
    closed = function.__closure__[0]
    original_class, original_code = closed.cell_contents, function.__code__
    closed.cell_contents = value.classes[wrong_owner]
    try:
        assert _fingerprint(function.__code__) == _fingerprint(primary_objects.codes['FullAttentionManager.cache_blocks'])
        with pytest.raises(ValueError, match='defining-class closure'):
            value.observer.attach()
        assert_originals_restored(value)
        assert function.__code__ is original_code and closed.cell_contents is value.classes[wrong_owner]
        assert value.observer._native_sources is None
    finally:
        closed.cell_contents = original_class
        value.mapping._cleanup_hooks()


def test_hash_mismatched_defining_source_refuses_and_does_not_repair_source(primary_objects, tmp_path):
    value = environment(primary_objects, tmp_path)
    path = tmp_path / _PATHS['Scheduler']
    altered = path.read_bytes() + b'\n# inert unapproved source bytes\n'
    path.write_bytes(altered)
    with pytest.raises(ValueError, match='source pin differs'):
        value.observer.attach()
    assert_originals_restored(value)
    assert path.read_bytes() == altered and value.observer._native_sources is None


def test_unproved_native_closure_is_unsupported_without_invoking_it(primary_objects, tmp_path):
    target = 'FullAttentionManager.cache_blocks'
    def alter(classes, natives):
        unproved = object()
        def replacement(self, *args, **kwargs):
            return unproved
        natives[target] = replacement
        classes['FullAttentionManager'].cache_blocks = replacement
    value = environment(primary_objects, tmp_path, alter)
    changed = value.natives[target]
    code = changed.__code__
    with pytest.raises(ValueError):
        value.observer.attach()
    assert_originals_restored(value)
    assert value.natives[target] is changed and changed.__code__ is code
    assert changed.__code__.co_freevars == ('unproved',)
    assert value.observer._native_sources is None and value.observer.records == []
