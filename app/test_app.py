import pytest
from fastapi.testclient import TestClient
import subprocess
import time
import os
import signal
import sys

import requests

import threading

import socket
import random

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
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(__file__) + "/.."
    proc = subprocess.Popen([
        "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"
    ], cwd=os.path.dirname(__file__) + "/..", env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Print the subprocess stdout in a separate thread
    def print_stdout(pipe):
        for line in iter(pipe.readline, b''):
            with open("test_app_uvicorn.log", "ab") as logfile:
                logfile.write(line)
                logfile.flush()

    threading.Thread(target=print_stdout, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=print_stdout, args=(proc.stderr,), daemon=True).start()
    wait_for_port(8000)
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

@pytest.fixture(scope="session")
def client(fastapi_app):
    base_url = "http://127.0.0.1:8000"
    return base_url

def test_tools_loaded(client):
    r = requests.get(f"{client}/tools")
    assert r.status_code == 200
    tools = r.json()
    tool_names = [t["name"] for t in tools]
    assert "add" in tool_names
    assert "hello" in tool_names
    assert "error" in tool_names

def test_call_tool(client):
    r = requests.post(f"{client}/tools/add", json={"a": 2, "b": 3})
    print(f"Response: {r.text}")
    assert r.status_code == 200
    assert r.json()["structuredContent"]["result"] == 5

def test_parallel_calls(client):
    def call_add():
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        r = requests.post(f"{client}/tools/add", json={"a": a, "b": b})
        assert r.status_code == 200
        assert r.json()["structuredContent"]["result"] == a + b
    threads = [threading.Thread(target=call_add) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # If no exceptions, parallel worked

def test_non_existing_tool(client):
    r = requests.post(f"{client}/tools/doesnotexist", json={})
    assert r.status_code == 404

def test_counts_up_in_a_session_correctly(client):
    r = requests.post(f"{client}/session/start")
    assert r.status_code == 200
    session_id = r.json()["x-inxm-mcp-session"]
    cookie = r.cookies.get_dict()
    r = requests.post(f"{client}/tools/call_counter", headers={"x-inxm-mcp-session": session_id}, json={})
    assert r.status_code == 200
    assert r.json()["structuredContent"]["result"] == 1
    
    # session via header
    r = requests.post(f"{client}/tools/call_counter", headers={"x-inxm-mcp-session": session_id}, json={})
    assert r.status_code == 200
    assert r.json()["structuredContent"]["result"] == 2
    
    # call without session should start a new session
    r = requests.post(f"{client}/tools/call_counter", json={})
    assert r.status_code == 200
    assert r.json()["structuredContent"]["result"] == 1
    
    # session via cookie
    r = requests.post(f"{client}/tools/call_counter", cookies=cookie, json={})
    assert r.status_code == 200
    assert r.json()["structuredContent"]["result"] == 3

def test_tool_wrong_args(client):
    r = requests.post(f"{client}/tools/add", json={"a": 1})
    assert r.status_code >= 400 and r.status_code < 500

def test_tool_error(client):
    r = requests.post(f"{client}/tools/error", json={"message": "fail!"})
    assert r.status_code >= 500
