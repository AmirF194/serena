import time

import pytest

from serena.task_executor import TaskExecutor
from serena.tools.tools_base import Tool, ToolCallError, ToolMarkerDoesNotRequireActiveProject


class _FakeSerenaConfig:
    def __init__(self, tool_timeout: float):
        self.tool_timeout = tool_timeout


class _FakeAgent:
    """
    A minimal stand-in for `SerenaAgent`, providing only what `Tool.apply_ex` needs,
    so that its timeout handling can be tested without a real project/language server.
    """

    def __init__(self, tool_timeout: float):
        self.serena_config = _FakeSerenaConfig(tool_timeout)
        self._task_executor = TaskExecutor("FakeAgentExecutor")

    def issue_task(self, task, name=None, logged=True, timeout=None):
        return self._task_executor.issue_task(task, name=name, logged=logged, timeout=timeout)

    def tool_is_active(self, tool_name: str) -> bool:
        return True

    def get_active_tool_names(self):
        return [SlowTool.get_name_from_cls()]

    def record_tool_usage(self, kwargs, result, tool) -> None:
        pass

    def get_language_server_manager(self):
        return None


class SlowTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """A tool whose `apply` sleeps longer than the configured timeout, to exercise apply_ex's TimeoutError handling."""

    def apply(self, delay: float) -> str:
        """
        :param delay: how long to sleep, in seconds
        """
        time.sleep(delay)
        return "done"


@pytest.fixture
def slow_tool():
    agent = _FakeAgent(tool_timeout=0.2)
    return SlowTool(agent)


def test_apply_ex_timeout_message_is_self_explanatory(slow_tool):
    """
    A tool call that exceeds the configured timeout should return a message that names the tool
    and the timeout, not the bare "TimeoutError: " that `str(TimeoutError())` produces.
    """
    result = slow_tool.apply_ex(log_call=False, catch_exceptions=True, delay=2)
    assert result != "TimeoutError: "
    assert SlowTool.get_name_from_cls() in result
    assert "0.2" in result
    assert "timed out" in result


def test_apply_ex_timeout_raises_tool_call_error_when_not_caught(slow_tool):
    with pytest.raises(ToolCallError) as exc_info:
        slow_tool.apply_ex(log_call=False, catch_exceptions=False, delay=2)
    msg = exc_info.value.get_error_message()
    assert msg != "TimeoutError: "
    assert SlowTool.get_name_from_cls() in msg
