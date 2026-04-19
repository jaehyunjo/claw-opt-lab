"""Verify the shim provides mocks for deleted modules and patches the registry."""

from __future__ import annotations

import json

from tools.registry import registry


def _default_value_for(prop: dict):
    """Return a minimal value that satisfies a schema property's type."""
    t = prop.get("type")
    if isinstance(t, list):
        t = t[0]
    return {
        "string": "x",
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "array": [],
        "object": {},
    }.get(t, "x")


def _valid_args(schema: dict) -> dict:
    params = schema.get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or []
    return {name: _default_value_for(props.get(name, {})) for name in required}


def test_tool_registry_populated():
    """After shim + discovery, the registry should contain a useful number of tools."""
    names = registry.get_all_tool_names()
    assert len(names) >= 20, f"Expected >=20 tools, got {len(names)}: {names}"


def test_deleted_modules_import_as_magicmock():
    """Modules we deleted from disk still import (as MagicMock) for tool files that reference them."""
    import agent
    import gateway
    import hermes_cli
    import model_tools  # noqa: F401
    assert agent is not None
    assert gateway is not None
    assert hermes_cli is not None


def test_deleted_submodules_import():
    """Deep paths under mocked packages should resolve to MagicMock."""
    from agent.auxiliary_client import call_llm  # noqa: F401
    from tools.environments.local import LocalEnvironment  # noqa: F401
    # If imports succeed without ImportError we're good — values are MagicMock.


def test_mock_handler_rejects_missing_required():
    """Every registered tool with required fields must error when we omit them."""
    checked = 0
    for name in registry.get_all_tool_names():
        entry = registry.get_entry(name)
        required = (entry.schema.get("parameters") or {}).get("required") or []
        if not required:
            continue
        result = json.loads(registry.dispatch(name, {}))
        assert "error" in result, f"{name}: dispatch with no args should error"
        assert result.get("mock") is True
        checked += 1
    assert checked > 0, "expected at least one tool with required fields"


def test_mock_handler_accepts_valid_input():
    """Every registered tool should succeed when fed schema-correct args."""
    for name in registry.get_all_tool_names():
        entry = registry.get_entry(name)
        args = _valid_args(entry.schema)
        raw = registry.dispatch(name, args)
        parsed = json.loads(raw)
        assert parsed.get("ok") is True, (
            f"{name}: expected ok, got {parsed} for args={args}"
        )
        assert parsed.get("mock") is True
        assert parsed.get("tool") == name


def test_mock_handler_rejects_wrong_type():
    """Passing the wrong JSON type for a required string property should error."""
    for name in registry.get_all_tool_names():
        entry = registry.get_entry(name)
        params = entry.schema.get("parameters") or {}
        required = params.get("required") or []
        props = params.get("properties") or {}
        # Find a required string prop, pass an int instead
        string_required = [r for r in required if props.get(r, {}).get("type") == "string"]
        if not string_required:
            continue
        bad_name = string_required[0]
        args = {r: _default_value_for(props.get(r, {})) for r in required}
        args[bad_name] = 12345  # wrong type
        parsed = json.loads(registry.dispatch(name, args))
        assert "error" in parsed, f"{name}: type mismatch on {bad_name} should error ({args})"
        return  # one positive case is enough
