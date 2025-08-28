import random
import threading


class FastAPIWrapper:
    def __init__(self, client, base_url=""):
        self.client = client
        self.base_url = base_url

    def test_tools_loaded(self):
        r = self.client.get(f"{self.base_url}/tools", timeout=10)
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        tools = r.json()
        tool_names = [t["name"] for t in tools]
        assert "add" in tool_names, f"'add' not in tools: {tool_names}"
        assert "hello" in tool_names, f"'hello' not in tools: {tool_names}"
        assert "error" in tool_names, f"'error' not in tools: {tool_names}"

    def test_prompts_loaded(self):
        r = self.client.get(f"{self.base_url}/prompts", timeout=10)
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        prompts = r.json()
        prompt_names = [p["name"] for p in prompts.get("prompts")]
        assert "greeting" in prompt_names, f"'greeting' not in prompts: {prompt_names}"

    def test_call_tool(self):
        r = self.client.post(
            f"{self.base_url}/tools/add", json={"a": 2, "b": 3}, timeout=10
        )
        print(f"Response: {r.text}")
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 5
        ), f"Unexpected result: {r.json()}"

    def test_call_prompt(self):
        r = self.client.post(
            f"{self.base_url}/prompts/greeting", json={"name": "John"}, timeout=10
        )
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert len(r.json()["messages"]) > 0, f"No messages: {r.json()}"
        assert (
            r.json()["messages"][0]["content"]["text"] == "Hello, John!"
        ), f"Unexpected result: {r.json()}"

    def test_tool_wrong_args(self):
        r = self.client.post(f"{self.base_url}/tools/add", json={"a": 1}, timeout=10)
        assert (
            r.status_code >= 400 and r.status_code < 500
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_tool_error(self):
        r = self.client.post(
            f"{self.base_url}/tools/error", json={"message": "fail!"}, timeout=10
        )
        assert (
            r.status_code >= 500
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_parallel_calls(self):
        def call_add():
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            r = self.client.post(
                f"{self.base_url}/tools/add", json={"a": a, "b": b}, timeout=10
            )
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
        r = self.client.post(f"{self.base_url}/tools/doesnotexist", json={}, timeout=10)
        assert (
            r.status_code == 404
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_prompts_in_a_session_doesnt_do_anything_but_it_should_work(self):
        r = self.client.post(f"{self.base_url}/session/start", timeout=10)
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

        r = self.client.get(f"{self.base_url}/prompts", timeout=10)
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        prompts = r.json()
        prompt_names = [p["name"] for p in prompts.get("prompts")]
        assert "greeting" in prompt_names, f"'greeting' not in prompts: {prompt_names}"

        r = self.client.post(
            f"{self.base_url}/prompts/greeting", json={"name": "John"}, timeout=10
        )
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert len(r.json()["messages"]) > 0, f"No messages: {r.json()}"
        assert (
            r.json()["messages"][0]["content"]["text"] == "Hello, John!"
        ), f"Unexpected result: {r.json()}"

        r = self.client.post(f"{self.base_url}/session/close", timeout=10)
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

    def test_counts_up_in_a_session_correctly(self):
        r = self.client.post(f"{self.base_url}/session/start", timeout=10)
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
            timeout=10,
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
            timeout=10,
        )
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 2
        ), f"Unexpected result: {r.json()}"

        self.client.cookies.clear()
        r = self.client.post(f"{self.base_url}/tools/call_counter", json={}, timeout=10)
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 1
        ), f"Unexpected result: {r.json()}"

        self.client.cookies.update(cookie)
        r = self.client.post(f"{self.base_url}/tools/call_counter", json={}, timeout=10)
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
        assert (
            r.json()["structuredContent"]["result"] == 3
        ), f"Unexpected result: {r.json()}"

        r = self.client.post(f"{self.base_url}/session/close", timeout=10)
        assert (
            r.status_code == 200
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"

        r = self.client.post(f"{self.base_url}/tools/call_counter", json={}, timeout=10)
        assert (
            r.status_code == 404
        ), f"Unexpected status code: {r.status_code}, Response: {r.text}"
