"""Minimal skeleton for Hermes LLM-routing experiments.

Load order:
    1. core.shim.install_shim()     — MagicMock deleted internal packages + heavy deps;
                                       patch tool registry so every registered
                                       tool handler becomes a mock.
    2. tools.registry.discover_builtin_tools()  — import all self-registering tool
                                                    files (now mock-handled).
    3. core.agent.run_conversation() — drive the agent loop against Azure OpenAI.
"""
