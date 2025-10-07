import pytest
from fastapi.testclient import TestClient

from app.utils_tests.fastapi_wrapper import FastAPIWrapper


@pytest.fixture(scope="session")
def test_client():
    from app.server import app

    with TestClient(app) as client:
        yield client


# Instantiate and run the TestClient-based tests
def test_tools_loaded(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_tools_loaded()


def test_prompts_loaded(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_prompts_loaded()


def test_call_prompt(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_call_prompt()


def test_call_tool(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_call_tool()


def test_tool_wrong_args(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_tool_wrong_args()


def test_tool_error(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_tool_error()


def test_parallel_calls(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_parallel_calls()


def test_non_existing_tool(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_non_existing_tool()


def test_counts_up_in_a_session_correctly(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_counts_up_in_a_session_correctly()


def test_prompts_in_a_session_doesnt_do_anything_but_it_should_work(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_prompts_in_a_session_doesnt_do_anything_but_it_should_work()


# Additional edge/error branch tests for routes.py
def test_start_session_group_access_denied(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_start_session_group_access_denied()


def test_run_tool_session_not_found(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_run_tool_session_not_found()


def test_run_prompt_session_not_found(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_run_prompt_session_not_found()


def test_run_tool_validation_error(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_run_tool_validation_error()


def test_run_prompt_validation_error(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_run_prompt_validation_error()


def test_run_tool_internal_error(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_run_tool_internal_error()


def test_run_prompt_internal_error(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_run_prompt_internal_error()


# Edge case tests for routes.py
def test_missing_session_header_on_close(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_missing_session_header_on_close()


def test_run_tool_missing_args(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_run_tool_missing_args()


def test_run_prompt_missing_args(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_run_prompt_missing_args()


def test_start_session_with_group_and_no_token(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_start_session_with_group_and_no_token()


def test_start_session_with_invalid_token(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_start_session_with_invalid_token()


def test_call_list_resources(test_client):
    suite = FastAPIWrapper(test_client)
    suite.test_call_list_resources()
