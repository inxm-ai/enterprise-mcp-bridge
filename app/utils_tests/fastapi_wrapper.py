import random
import threading


class FastAPIWrapper:
    def __init__(self, client, base_url=""):
        self.client = client
        self.base_url = base_url

    def test_tools_loaded(self):
        r = self.client.get(f"{self.base_url}/tools", timeout=10)
        assert r.status_code == 200
        tools = r.json()
        tool_names = [t["name"] for t in tools]
        assert "add" in tool_names
        assert "hello" in tool_names
        assert "error" in tool_names

    def test_call_tool(self):
        r = self.client.post(
            f"{self.base_url}/tools/add", json={"a": 2, "b": 3}, timeout=10
        )
        print(f"Response: {r.text}")
        assert r.status_code == 200
        assert r.json()["structuredContent"]["result"] == 5

    def test_tool_wrong_args(self):
        r = self.client.post(f"{self.base_url}/tools/add", json={"a": 1}, timeout=10)
        assert r.status_code >= 400 and r.status_code < 500

    def test_tool_error(self):
        r = self.client.post(
            f"{self.base_url}/tools/error", json={"message": "fail!"}, timeout=10
        )
        assert r.status_code >= 500

    def test_parallel_calls(self):
        def call_add():
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            r = self.client.post(
                f"{self.base_url}/tools/add", json={"a": a, "b": b}, timeout=10
            )
            assert r.status_code == 200
            assert r.json()["structuredContent"]["result"] == a + b

        threads = [threading.Thread(target=call_add) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # If no exceptions, parallel worked

    def test_non_existing_tool(self):
        r = self.client.post(f"{self.base_url}/tools/doesnotexist", json={}, timeout=10)
        assert r.status_code == 404

    def test_counts_up_in_a_session_correctly(self):
        r = self.client.post(f"{self.base_url}/session/start", timeout=10)
        assert r.status_code == 200
        session_id = r.json()["x-inxm-mcp-session"]

        if hasattr(r.cookies, "get_dict"):
            cookie = r.cookies.get_dict()
        else:
            cookie = dict(r.cookies)

        assert "x-inxm-mcp-session" in cookie
        r = self.client.post(
            f"{self.base_url}/tools/call_counter",
            headers={"x-inxm-mcp-session": session_id},
            json={},
            timeout=10,
        )
        assert r.status_code == 200
        assert r.json()["structuredContent"]["result"] == 1

        # session via header
        r = self.client.post(
            f"{self.base_url}/tools/call_counter",
            headers={"x-inxm-mcp-session": session_id},
            json={},
            timeout=10,
        )
        assert r.status_code == 200
        assert r.json()["structuredContent"]["result"] == 2

        self.client.cookies.clear()
        r = self.client.post(f"{self.base_url}/tools/call_counter", json={}, timeout=10)
        assert r.status_code == 200
        assert r.json()["structuredContent"]["result"] == 1

        self.client.cookies.update(cookie)
        r = self.client.post(f"{self.base_url}/tools/call_counter", json={}, timeout=10)
        assert r.status_code == 200
        assert r.json()["structuredContent"]["result"] == 3
