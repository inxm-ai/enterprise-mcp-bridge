from app.tgi.services.tool_service import ToolService


class TestToolServiceErrorDetection:
    def test_result_has_error_valid_json_with_error_string(self):
        """Test that valid JSON containing 'error' string but no error structure is NOT flagged as error."""
        service = ToolService()

        # This mimics the user's case: get_servers returning a list of servers,
        # where one might have "error" in the name or description, but it's a valid success response.
        result = {
            "content": '[{"name": "server-with-error-handling", "status": "running"}]',
            "role": "tool",
            "tool_call_id": "123",
            "name": "get_servers",
        }

        assert service._result_has_error(result) is False

    def test_result_has_error_valid_json_with_error_key_in_dict(self):
        """Test that valid JSON with 'error' key IS flagged as error."""
        service = ToolService()

        result = {
            "content": '{"error": "Something went wrong"}',
            "role": "tool",
            "tool_call_id": "123",
            "name": "get_servers",
        }

        assert service._result_has_error(result) is True

    def test_result_has_error_valid_json_list_with_error_key(self):
        """Test that valid JSON list containing an item with 'error' key IS flagged as error."""
        service = ToolService()

        result = {
            "content": '[{"error": "Failed to connect"}]',
            "role": "tool",
            "tool_call_id": "123",
            "name": "get_servers",
        }

        assert service._result_has_error(result) is True

    def test_result_has_error_plain_string_with_error(self):
        """Test that plain string containing 'error' IS flagged as error (fallback behavior)."""
        service = ToolService()

        result = {
            "content": "Error: could not fetch servers",
            "role": "tool",
            "tool_call_id": "123",
            "name": "get_servers",
        }

        assert service._result_has_error(result) is True

    def test_result_has_error_plain_string_without_error(self):
        """Test that plain string without 'error' is NOT flagged."""
        service = ToolService()

        result = {
            "content": "Everything is fine",
            "role": "tool",
            "tool_call_id": "123",
            "name": "get_servers",
        }

        assert service._result_has_error(result) is False

    def test_result_has_error_is_error_flag(self):
        """Test that isError flag triggers error detection."""
        service = ToolService()

        result = {"isError": True, "content": "some content"}

        assert service._result_has_error(result) is True

    def test_result_has_error_success_false_flag(self):
        """Test that success=False flag triggers error detection."""
        service = ToolService()

        result = {"success": False, "content": "some content"}

        assert service._result_has_error(result) is True

    def test_result_has_error_nested_json_text_error(self):
        """Test that JSON error embedded in content[].text is detected."""
        service = ToolService()
        result = {
            "content": '[{"text":"{\\"error\\":\\"No coordinates found\\"}"}]',
            "role": "tool",
            "tool_call_id": "123",
            "name": "weather_lookup",
        }
        assert service._result_has_error(result) is True

    def test_result_has_error_nested_json_text_success(self):
        """Test that JSON success payload embedded in content[].text is not flagged."""
        service = ToolService()
        result = {
            "content": '[{"text":"{\\"payload\\":{\\"plan_id\\":\\"plan-1\\"}}"}]',
            "role": "tool",
            "tool_call_id": "123",
            "name": "plan_lookup",
        }
        assert service._result_has_error(result) is False
