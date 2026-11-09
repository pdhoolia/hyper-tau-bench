"""The compact run result (``--result-json``) carries each test's conversation id."""

from types import SimpleNamespace

from tau2.hyper.data_model import OuterLoopResult
from tau2.hyper.result_serialization import compact_run_result


def test_compact_run_result_keeps_each_tests_conversation_id():
    result = OuterLoopResult.model_construct(
        final_test_reward=0.5,
        final_quality_reward=0.5,
        performance_reward=1.0,
        performance_penalty=0.0,
        total_outer_steps=3,
        run_metadata={"builder": {"harness": {"name": "claude-code"}}},
        test_details=[
            {"task_id": "0", "reward": 1.0, "error": None, "agent_credit_usage": None, "conversation_id": "hyper-r-arm-a-test-0"},
            {"task_id": "1", "reward": 0.0, "error": None, "agent_credit_usage": None, "conversation_id": None},
            {"task_id": "2", "reward": 0.0, "error": None, "agent_credit_usage": None},
        ],
    )
    compact = compact_run_result(
        task=SimpleNamespace(id="007", source_domain="retail_plus"),
        result=result,
        config={"developer_harness": "claude-code", "developer_llm": "m", "test_task_limit": 3},
        recording_path="rec.json",
    )
    assert [d["conversation_id"] for d in compact["test_details"]] == ["hyper-r-arm-a-test-0", None, None]
    assert [d["task_id"] for d in compact["test_details"]] == ["0", "1", "2"]
