import json

import pytest

from dibo.llm_update import (
    FileRoundtripReviewer,
    ReviewAbortedError,
    process_review_request,
    read_review_response,
    write_review_request,
)
from tests.unit.test_llm_update import evidence_fixture, proposal


def test_file_bridge_uses_same_schema_and_preserves_request_identity(tmp_path) -> None:
    evidence = evidence_fixture()
    request_path = write_review_request(tmp_path, "request_013", evidence)
    response_path = process_review_request(
        request_path,
        tmp_path / "responses",
        lambda payload: proposal().model_dump(mode="json"),
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))

    assert request["request_id"] == response["request_id"] == "request_013"
    assert request["parent_action_version"] == response["parent_action_version"] == 1
    assert (
        read_review_response(response_path, evidence, request_id="request_013").decision == "modify"
    )


def test_worker_timeout_writes_error_and_controller_keeps(tmp_path) -> None:
    evidence = evidence_fixture()
    request_path = write_review_request(tmp_path, "request_timeout", evidence)

    def timeout(payload):
        raise TimeoutError

    response_path = process_review_request(request_path, tmp_path / "responses", timeout)

    assert json.loads(response_path.read_text(encoding="utf-8"))["status"] == "error"
    assert (
        read_review_response(response_path, evidence, request_id="request_timeout").decision
        == "keep"
    )


def test_processed_request_never_calls_reviewer_twice(tmp_path) -> None:
    request = write_review_request(tmp_path, "once", evidence_fixture())
    process_review_request(
        request, tmp_path / "responses", lambda payload: proposal().model_dump(mode="json")
    )
    with pytest.raises(FileExistsError, match="replay"):
        process_review_request(
            request, tmp_path / "responses", lambda payload: pytest.fail("replayed API")
        )


def test_file_roundtrip_reviewer_waits_for_matching_response(tmp_path) -> None:
    evidence = evidence_fixture()
    calls = 0

    def sleep(_seconds):
        nonlocal calls
        calls += 1
        process_review_request(
            tmp_path / "llm_io/requests/fixed.json",
            tmp_path / "llm_io/responses",
            lambda payload: proposal().model_dump(mode="json"),
        )

    reviewer = FileRoundtripReviewer(
        tmp_path / "llm_io",
        timeout_s=2,
        poll_interval_s=0.1,
        request_id_factory=lambda evidence: "fixed",
        sleep=sleep,
    )
    update = reviewer(evidence.model_dump(mode="json"))

    assert calls == 1
    assert update["decision"] == "modify"


def test_file_roundtrip_timeout_writes_wait_heartbeats(tmp_path) -> None:
    evidence = evidence_fixture()
    now = 0.0

    def clock():
        return now

    def sleep(seconds):
        nonlocal now
        now += seconds

    reviewer = FileRoundtripReviewer(
        tmp_path / "llm_io",
        timeout_s=3,
        poll_interval_s=1,
        heartbeat_s=1,
        request_id_factory=lambda evidence: "timeout",
        sleep=sleep,
        clock=clock,
    )
    with pytest.raises(TimeoutError, match="timed out"):
        reviewer(evidence.model_dump(mode="json"))

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "llm_request_published"
    assert sum(event["event"] == "llm_waiting" for event in events) >= 2
    assert events[-1]["event"] == "llm_wait_timeout"


def test_stop_file_aborts_file_roundtrip_without_waiting_for_timeout(tmp_path) -> None:
    evidence = evidence_fixture()
    (tmp_path / "STOP").write_text("operator cancelled review\n")
    reviewer = FileRoundtripReviewer(
        tmp_path / "llm_io",
        timeout_s=30,
        request_id_factory=lambda evidence: "stop",
    )

    with pytest.raises(ReviewAbortedError, match="operator requested"):
        reviewer(evidence.model_dump(mode="json"))

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "llm_wait_aborted"
    assert events[-1]["message"] == "operator cancelled review"
