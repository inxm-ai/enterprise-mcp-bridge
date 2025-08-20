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
