"""
Tests for RustAnalyzer's ContentModified retry wrapper (#1724).

rust-analyzer turns a salsa computation cancelled by a concurrent workspace mutation into a
ContentModified (-32801) error response for methods -- like hover -- outside its own hard-coded
internal-retry set, so recovery is the client's job. These tests exercise
RustAnalyzer._install_content_modified_retry() in isolation, injecting deterministic
ContentModified/other-error responses instead of relying on the flaky Windows CI race that
motivated the fix.
"""

import pytest

from solidlsp.language_servers.rust_analyzer import RustAnalyzer
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.lsp_protocol_handler.lsp_types import LSPErrorCodes
from solidlsp.lsp_protocol_handler.server import LSPError


class _FakeServer:
    """Minimal stand-in for the LanguageServerInterface, exposing only send_request."""

    def __init__(self, responses):
        # Each entry is either a payload to return or an Exception instance to raise.
        self._responses = list(responses)
        self.calls = []

    def send_request(self, method, params=None):
        self.calls.append((method, params))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _make_analyzer(server) -> RustAnalyzer:
    analyzer = object.__new__(RustAnalyzer)
    analyzer.server = server
    return analyzer


def _content_modified_exception() -> SolidLSPException:
    return SolidLSPException("content modified", cause=LSPError(LSPErrorCodes.ContentModified, "content modified"))


@pytest.mark.rust
class TestRustAnalyzerContentModifiedRetry:
    def test_retries_and_succeeds_after_transient_content_modified(self):
        server = _FakeServer([_content_modified_exception(), _content_modified_exception(), {"contents": "ok"}])
        analyzer = _make_analyzer(server)

        analyzer._install_content_modified_retry(max_retries=5, retry_delay=0)
        result = analyzer.server.send_request("textDocument/hover", {"position": "irrelevant"})

        assert result == {"contents": "ok"}
        assert len(server.calls) == 3

    def test_exhausts_retries_and_raises_last_exception(self):
        exc = _content_modified_exception()
        server = _FakeServer([exc, exc, exc])
        analyzer = _make_analyzer(server)

        analyzer._install_content_modified_retry(max_retries=3, retry_delay=0)
        with pytest.raises(SolidLSPException) as exc_info:
            analyzer.server.send_request("textDocument/hover", {})

        assert exc_info.value is exc
        assert len(server.calls) == 3

    def test_non_content_modified_error_is_not_retried(self):
        other = SolidLSPException("boom", cause=LSPError(LSPErrorCodes.RequestFailed, "boom"))
        server = _FakeServer([other, {"contents": "unreachable"}])
        analyzer = _make_analyzer(server)

        analyzer._install_content_modified_retry(max_retries=5, retry_delay=0)
        with pytest.raises(SolidLSPException) as exc_info:
            analyzer.server.send_request("textDocument/hover", {})

        assert exc_info.value is other
        assert len(server.calls) == 1

    def test_request_cancelled_is_not_retried(self):
        """RequestCancelled (-32800) is client-initiated per the LSP spec; only
        ContentModified (-32801) is the transient server condition worth retrying (#1724).
        """
        cancelled = SolidLSPException("cancelled", cause=LSPError(LSPErrorCodes.RequestCancelled, "cancelled"))
        server = _FakeServer([cancelled, {"contents": "unreachable"}])
        analyzer = _make_analyzer(server)

        analyzer._install_content_modified_retry(max_retries=5, retry_delay=0)
        with pytest.raises(SolidLSPException) as exc_info:
            analyzer.server.send_request("textDocument/hover", {})

        assert exc_info.value is cancelled
        assert len(server.calls) == 1
