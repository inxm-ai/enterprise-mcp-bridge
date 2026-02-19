from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.elicitation import ElicitationRequiredError
from app.routes import run_tool


@pytest.mark.asyncio
async def test_tools_call_feedback_required_returns_409_with_canonical_payload():
    payload = {
        "message": "Pick a mode",
        "requestedSchema": {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["run_now", "schedule"]}},
            "required": ["mode"],
            "additionalProperties": False,
        },
        "meta": {},
    }

    with patch("app.routes.mcp_session_context") as mock_context:
        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = ElicitationRequiredError(
            payload, session_key="sess-1", requires_session=False
        )
        mock_context.return_value.__aenter__.return_value = mock_session

        with pytest.raises(HTTPException) as exc_info:
            await run_tool(
                "test_tool",
                request=SimpleNamespace(headers={}),
                x_inxm_mcp_session_header="sess-1",
                x_inxm_mcp_session_cookie=None,
                x_inxm_dry_run=None,
                access_token=None,
                args={},
                group=None,
            )

    assert exc_info.value.status_code == 409
    detail = exc_info.value.detail
    assert detail["error"] == "feedback_required"
    assert detail["awaiting_feedback"] is True
    assert detail["elicitation"]["message"] == "Pick a mode"
    assert detail["elicitation"]["requestedSchema"]["properties"]["mode"]["enum"] == [
        "run_now",
        "schedule",
    ]


@pytest.mark.asyncio
async def test_tools_call_feedback_resume_requires_pending_elicitation():
    coordinator = MagicMock()
    coordinator.submit_feedback.return_value = False

    with patch("app.routes.get_elicitation_coordinator", return_value=coordinator):
        with pytest.raises(HTTPException) as exc_info:
            await run_tool(
                "test_tool",
                request=SimpleNamespace(headers={}),
                x_inxm_mcp_session_header="sess-1",
                x_inxm_mcp_session_cookie=None,
                x_inxm_dry_run=None,
                access_token=None,
                args={"_user_feedback": "<user_feedback>run_now</user_feedback>"},
                group=None,
            )

    assert exc_info.value.status_code == 409
    detail = exc_info.value.detail
    assert detail["error"] == "feedback_not_expected"
