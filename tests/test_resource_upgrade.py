"""Regression: old Streamlit resources must upgrade on cache hits, not just creation."""
from threading import RLock
from types import SimpleNamespace
from document_app.service import LocalModels,upgrade_cached_models


def legacy():
    old=type('LocalModels',(),{})()
    old.lock=RLock();old._embedding=object();old._qa=object()
    return old


def test_pre_vision_resource_preserves_loaded_weights():
    old=legacy();weights=(old._embedding,old._qa,old.lock)
    assert not hasattr(old,'vision')
    upgraded=upgrade_cached_models(old)
    assert upgraded is old and isinstance(old,LocalModels)
    assert (old._embedding,old._qa,old.lock)==weights
    assert old.vision().model=='qwen3-vl:4b-instruct'
    assert upgrade_cached_models(old) is old


def test_global_resource_and_session_resource_upgrade_independently():
    global_resource=legacy();session_resource=legacy()
    session=SimpleNamespace(models=session_resource)
    session.models=upgrade_cached_models(session.models)
    # The bug upgraded only the session resource; readiness used a different one.
    assert not hasattr(global_resource,'vision')
    cached_readiness_resource=upgrade_cached_models(global_resource)
    assert callable(cached_readiness_resource.vision)
    assert callable(session.models.vision)
    assert cached_readiness_resource is not session.models


def test_app_starts_with_two_different_legacy_cached_resources(monkeypatch):
    from pathlib import Path
    from types import ModuleType
    import sys
    from streamlit.testing.v1 import AppTest
    from document_app.service import DocumentSession
    from document_app.vision import LocalVision
    module=ModuleType('legacy_fixture_resource')
    module.old_resource=legacy()
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setattr(LocalVision,'check',lambda self:'local ready')
    source=(Path(__file__).resolve().parents[1]/'app.py').read_text()
    source=source.replace('return LocalModels()','return old_resource')
    source=source.replace('def _cached_models(version=4):','def _cached_models(version=8675309):')
    source=source.replace('def vision_readiness():','def legacy_test_readiness():').replace('vision_readiness()', 'legacy_test_readiness()').replace('vision_readiness.clear()', 'legacy_test_readiness.clear()')
    # from_file normally supplies __file__, needed for the existing CSS loader.
    source='from legacy_fixture_resource import old_resource\n__file__='+repr(str(Path(__file__).resolve().parents[1]/'app.py'))+'\n'+source
    session=DocumentSession(legacy())
    try:
        at=AppTest.from_string(source,default_timeout=120)
        at.session_state['documents']=session
        at.run()
        assert not at.exception and len(at.file_uploader)==1
        assert callable(module.old_resource.vision)
        assert callable(session.models.vision)
        at.run()  # Resource cache hit must also be safe.
        assert not at.exception
    finally:session.close()
