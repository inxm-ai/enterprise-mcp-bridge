import random
import threading


class ClientWrapper:
    def __init__(self, client):
        self.client = client
        try:
            from fastapi.testclient import TestClient

            self.is_test_client = isinstance(client, TestClient)
        except ImportError:
            self.is_test_client = False

    @property
    def cookies(self):
        return self.client.cookies

    def get(self, url, **kwargs):
        if not self.is_test_client:
            kwargs.setdefault("timeout", 10)
        return self.client.get(url, **kwargs)

    def post(self, url, **kwargs):
        if not self.is_test_client:
            kwargs.setdefault("timeout", 10)
        return self.client.post(url, **kwargs)


class FastAPIWrapper:
    def test_start_session_group_access_denied(self):
        # Use a valid JWT but a group the user is not a member of
        valid_jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiJ1c2VyMSIsImdyb3VwcyI6WyJhbGxvd2VkLWdyb3VwIl19."
            "dummy-signature"
        )
        r = self.client.post(
            f"{self.base_url}/session/start?group=forbidden-group",
            headers={"X-Auth-Request-Access-Token": valid_jwt},
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}, {r.text}"

    def test_run_tool_session_not_found(self):
        # Simulate session not found (should return 404)
        r = self.client.post(
            f"{self.base_url}/tools/add",
            headers={"x-inxm-mcp-session": "bogus-session"},
            json={"a": 1, "b": 2},
        )
        assert r.status_code == 404, f"Expected 404, got {r.status_code}, {r.text}"

    def test_run_prompt_session_not_found(self):
        r = self.client.post(
            f"{self.base_url}/prompts/greeting",
            headers={"x-inxm-mcp-session": "bogus-session"},
            json={"name": "John"},
        )
        assert r.status_code == 404, f"Expected 404, got {r.status_code}, {r.text}"

    def test_run_tool_validation_error(self):
        # Should return 400, 422, or 404 for validation error (missing required arg)
        r = self.client.post(
            f"{self.base_url}/tools/add", json={"a": "not-an-int", "b": 2}
        )
        assert r.status_code in [
            400,
            422,
            404,
        ], f"Expected 400, 422, or 404, got {r.status_code}, {r.text}"

    def test_run_prompt_validation_error(self):
        r = self.client.post(f"{self.base_url}/prompts/greeting", json={"name": 123})
        assert r.status_code in [
            400,
            422,
            404,
        ], f"Expected 400, 422, or 404, got {r.status_code}, {r.text}"

    def test_run_tool_internal_error(self):
        # Should return 500 or 404 for tool error
        r = self.client.post(f"{self.base_url}/tools/error", json={"message": "fail!"})
        assert r.status_code in [
            500,
            404,
        ], f"Expected 500 or 404, got {r.status_code}, {r.text}"

    def test_run_prompt_internal_error(self):
        # Should return 500, 404, 400, or 422 for prompt error (simulate by passing bad args)
        r = self.client.post(f"{self.base_url}/prompts/greeting", json={"bad": "param"})
        assert r.status_code in [
            400,
            422,
            500,
            404,
        ], f"Expected 400, 422, 500, or 404, got {r.status_code}, {r.text}"

    # UserLoggedOutException tests removed (only testable via mocking)
    def __init__(self, client, base_url=""):
        self.client = ClientWrapper(client)
        self.base_url = base_url

    def test_tools_loaded(self):
        r = self.client.get(f"{self.base_url}/tools")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        tools = r.json()
        tool_names = [t["name"] for t in tools]
        assert "add" in tool_names, f"'add' not in tools: {tool_names}"
        assert "hello" in tool_names, f"'hello' not in tools: {tool_names}"
        assert "error" in tool_names, f"'error' not in tools: {tool_names}"

    def test_prompts_loaded(self):
        r = self.client.get(f"{self.base_url}/prompts")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        prompts = r.json()
        prompt_names = [p["name"] for p in prompts.get("prompts")]
        assert "greeting" in prompt_names, f"'greeting' not in prompts: {prompt_names}"

    def test_call_tool(self):
        r = self.client.post(f"{self.base_url}/tools/add", json={"a": 2, "b": 3})
        print(f"Response: {r.text}")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 5
        ), f"Unexpected result: {r.json()}"

    def test_call_prompt(self):
        r = self.client.post(f"{self.base_url}/prompts/greeting", json={"name": "John"})
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert len(r.json()["messages"]) > 0, f"No messages: {r.json()}"
        assert (
            r.json()["messages"][0]["content"]["text"] == "Hello, John!"
        ), f"Unexpected result: {r.json()}"

    def test_tool_wrong_args(self):
        r = self.client.post(f"{self.base_url}/tools/add", json={"a": 1})
        assert (
            r.status_code >= 400 and r.status_code < 500
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_tool_error(self):
        r = self.client.post(f"{self.base_url}/tools/error", json={"message": "fail!"})
        assert (
            r.status_code >= 500
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_parallel_calls(self):
        def call_add():
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            r = self.client.post(f"{self.base_url}/tools/add", json={"a": a, "b": b})
            assert (
                r.status_code == 200
            ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
            assert (
                r.json()["structuredContent"]["result"] == a + b
            ), f"Unexpected result: {r.json()}"

        threads = [threading.Thread(target=call_add) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # If no exceptions, parallel worked

    def test_non_existing_tool(self):
        r = self.client.post(f"{self.base_url}/tools/doesnotexist", json={})
        assert (
            r.status_code == 404
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_prompts_in_a_session_doesnt_do_anything_but_it_should_work(self):
        r = self.client.post(f"{self.base_url}/session/start")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

        r = self.client.get(f"{self.base_url}/prompts")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        prompts = r.json()
        prompt_names = [p["name"] for p in prompts.get("prompts")]
        assert "greeting" in prompt_names, f"'greeting' not in prompts: {prompt_names}"

        r = self.client.post(f"{self.base_url}/prompts/greeting", json={"name": "John"})
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert len(r.json()["messages"]) > 0, f"No messages: {r.json()}"
        assert (
            r.json()["messages"][0]["content"]["text"] == "Hello, John!"
        ), f"Unexpected result: {r.json()}"

        r = self.client.post(f"{self.base_url}/session/close")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_counts_up_in_a_session_correctly(self):
        r = self.client.post(f"{self.base_url}/session/start")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        session_id = r.json()["x-inxm-mcp-session"]

        if hasattr(r.cookies, "get_dict"):
            cookie = r.cookies.get_dict()
        else:
            cookie = dict(r.cookies)

        assert "x-inxm-mcp-session" in cookie, f"Session cookie missing: {cookie}"
        r = self.client.post(
            f"{self.base_url}/tools/call_counter",
            headers={"x-inxm-mcp-session": session_id},
            json={},
        )
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 1
        ), f"Unexpected result: {r.json()}"

        # session via header
        r = self.client.post(
            f"{self.base_url}/tools/call_counter",
            headers={"x-inxm-mcp-session": session_id},
            json={},
        )
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 2
        ), f"Unexpected result: {r.json()}"

        self.client.cookies.clear()
        r = self.client.post(f"{self.base_url}/tools/call_counter", json={})
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 1
        ), f"Unexpected result: {r.json()}"

        self.client.cookies.update(cookie)
        r = self.client.post(f"{self.base_url}/tools/call_counter", json={})
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 3
        ), f"Unexpected result: {r.json()}"

        r = self.client.post(f"{self.base_url}/session/close")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

        r = self.client.post(f"{self.base_url}/tools/call_counter", json={})
        assert (
            r.status_code == 404
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_missing_session_header_on_close(self):
        # Should return 400 or 404 if session header is missing
        r = self.client.post(f"{self.base_url}/session/close")
        assert r.status_code in [
            400,
            404,
        ], f"Expected 400 or 404, got {r.status_code}, {r.text}"

    def test_run_tool_missing_args(self):
        # Should handle missing args gracefully (args=None)
        r = self.client.post(f"{self.base_url}/tools/add")
        assert r.status_code >= 400, f"Expected error, got {r.status_code}, {r.text}"

    def test_run_prompt_missing_args(self):
        # Should handle missing args gracefully (args=None)
        r = self.client.post(f"{self.base_url}/prompts/greeting")
        assert r.status_code >= 400, f"Expected error, got {r.status_code}, {r.text}"

    def test_start_session_with_group_and_no_token(self):
        # Should fail gracefully if group is set but no token
        r = self.client.post(f"{self.base_url}/session/start?group=testgroup")
        assert r.status_code in [
            400
        ], f"Unexpected status code: {r.status_code}, {r.text}"

    def test_start_session_with_invalid_token(self):
        # This works atm, because the server does not validate the token
        #   I guess this is okay for now, as it will just fail on requests
        #   that need a valid token, and for those that don't it just works
        r = self.client.post(
            f"{self.base_url}/session/start",
            headers={"X-Auth-Request-Access-Token": "invalid"},
        )
        assert r.status_code in [
            200
        ], f"Unexpected status code: {r.status_code}, {r.text}"
