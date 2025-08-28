import pytest
import subprocess
import time
import os

import requests

import threading

import socket

from app.utils_tests.fastapi_wrapper import FastAPIWrapper


def wait_for_port(port, host="127.0.0.1", timeout=60.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Timeout waiting for {host}:{port}")


@pytest.fixture(scope="session")
def fastapi_app():
    # Start the FastAPI app as a subprocess
    port = 8888
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(__file__) + "/.."
    proc = subprocess.Popen(
        ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", f"{port}"],
        cwd=os.path.dirname(__file__) + "/..",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Print the subprocess stdout in a separate thread
    def print_stdout(pipe):
        for line in iter(pipe.readline, b""):
            with open("test_app_uvicorn.log", "ab") as logfile:
                logfile.write(line)
                logfile.flush()

    threading.Thread(target=print_stdout, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=print_stdout, args=(proc.stderr,), daemon=True).start()
    wait_for_port(port)
    yield proc, port
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def test_client(fastapi_app):

    base_url = f"http://127.0.0.1:{fastapi_app[1]}"
    session = requests.Session()
    return session, base_url


# Instantiate and run the TestClient-based tests
def test_tools_loaded(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_tools_loaded()


def test_prompts_loaded(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_prompts_loaded()


def test_call_tool(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_call_tool()


def test_call_prompt(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_call_prompt()


def test_tool_wrong_args(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_tool_wrong_args()


def test_tool_error(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_tool_error()


def test_parallel_calls(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_parallel_calls()


def test_non_existing_tool(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_non_existing_tool()


def test_counts_up_in_a_session_correctly(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_counts_up_in_a_session_correctly()


def test_prompts_in_a_session_doesnt_do_anything_but_it_should_work(test_client):
    client, base_url = test_client
    suite = FastAPIWrapper(client, base_url)
    suite.test_prompts_in_a_session_doesnt_do_anything_but_it_should_work()
