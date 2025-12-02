"""
Pytest tests for group-based data access functionality
"""

import pytest
import jwt
from datetime import datetime, timedelta, UTC
from unittest.mock import patch
from app.oauth.user_info import UserInfoExtractor, DataAccessManager


@pytest.fixture
def extractor():
    """Fixture for UserInfoExtractor"""
    return UserInfoExtractor()


@pytest.fixture
def data_manager():
    """Fixture for DataAccessManager"""
    return DataAccessManager()


def create_test_token(user_id: str, groups: list, email: str = None):
    """Create a test JWT token for testing purposes"""
    payload = {
        "sub": user_id,
        "email": email or f"{user_id}@test.com",
        "preferred_username": user_id,
        "name": f"Test User {user_id}",
        "groups": groups,
        "realm_access": {"roles": ["user"] + groups[:2]},  # Add some roles
        "iss": "http://test-keycloak/realms/test",
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }

    # Create unsigned token for testing
    return jwt.encode(payload, "secret", algorithm="HS256")


class TestUserInfoExtraction:
    """Test user info extraction from tokens"""

    def test_extract_user_info_basic(self, extractor):
        """Test basic user info extraction"""
        token = create_test_token(
            user_id="testuser123",
            groups=["finance", "marketing", "admin"],
            email="test.user@company.com",
        )

        user_info = extractor.extract_user_info(token)

        assert user_info["user_id"] == "testuser123"
        assert user_info["email"] == "test.user@company.com"
        assert "finance" in user_info["groups"]
        assert "marketing" in user_info["groups"]
        assert "admin" in user_info["groups"]

    def test_extract_user_info_with_roles(self, extractor):
        """Test user info extraction with realm roles"""
        token = create_test_token(
            user_id="roleuser", groups=["finance", "hr"], email="role.user@company.com"
        )

        user_info = extractor.extract_user_info(token)

        assert user_info["user_id"] == "roleuser"
        assert "user" in user_info["roles"]  # From realm_access.roles
        assert "finance" in user_info["roles"]  # First group added to roles

    def test_invalid_token_raises_error(self, extractor):
        """Test that invalid token raises appropriate error"""
        with pytest.raises(AssertionError, match="Invalid token format"):
            extractor.extract_user_info("invalid-token")


class TestDataAccessResolution:
    """Test data resource resolution based on group membership"""

    def test_user_specific_data_access(self, data_manager):
        """Test resolving user-specific data resource"""
        token = create_test_token("alice", ["finance", "employees"])

        resource = data_manager.resolve_data_resource(token)

        assert resource == "u/alice"

    def test_group_specific_data_access_allowed(self, data_manager):
        """Test resolving group-specific data resource when user is member"""
        token = create_test_token("alice", ["finance", "employees"])

        resource = data_manager.resolve_data_resource(token, "finance")

        assert resource == "g/finance"

    def test_group_specific_data_access_denied(self, data_manager):
        """Test access denial when user is not member of requested group"""
        token = create_test_token("bob", ["marketing", "employees"])

        with pytest.raises(
            PermissionError, match="User bob is not a member of group 'finance'"
        ):
            data_manager.resolve_data_resource(token, "finance")

    def test_admin_access_to_multiple_groups(self, data_manager):
        """Test admin user can access multiple groups"""
        admin_token = create_test_token("admin", ["finance", "marketing", "admin"])

        finance_resource = data_manager.resolve_data_resource(admin_token, "finance")
        marketing_resource = data_manager.resolve_data_resource(
            admin_token, "marketing"
        )

        assert finance_resource == "g/finance"
        assert marketing_resource == "g/marketing"


class TestIdentifierSanitization:
    """Test identifier sanitization for security"""

    def test_sanitize_basic_identifier(self, data_manager):
        """Test basic identifier sanitization"""
        sanitized = data_manager._sanitize_identifier("normal-group_name.test")
        assert sanitized == "normal-group_name.test"

    def test_sanitize_removes_dangerous_chars(self, data_manager):
        """Test removal of potentially dangerous characters"""
        dangerous = "group/with<>slashes&pipes|"
        sanitized = data_manager._sanitize_identifier(dangerous)

        # Should only contain safe characters
        assert "/" not in sanitized
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "&" not in sanitized
        assert "|" not in sanitized
        assert sanitized == "groupwithslashespipes"

    def test_sanitize_length_limit(self, data_manager):
        """Test length limitation"""
        long_identifier = "a" * 150
        sanitized = data_manager._sanitize_identifier(long_identifier)

        assert len(sanitized) <= 100

    def test_data_resource_with_sanitized_group(self, data_manager):
        """Test that group names are sanitized in resource resolution"""
        token = create_test_token("user", ["group/with/slashes"])

        # This should work because we sanitize the group name
        resource = data_manager.resolve_data_resource(token, "group/with/slashes")

        assert resource == "g/groupwithslashes"  # Slashes removed


class TestUserGroupMembership:
    """Test user group membership checking"""

    def test_user_in_direct_group(self, data_manager):
        """Test direct group membership"""
        user_info = {"groups": ["finance", "marketing"], "roles": []}

        assert data_manager._user_in_group(user_info, "finance") is True
        assert data_manager._user_in_group(user_info, "marketing") is True
        assert data_manager._user_in_group(user_info, "hr") is False

    def test_user_in_role_group(self, data_manager):
        """Test role-based group membership"""
        user_info = {"groups": [], "roles": ["admin", "user"]}

        assert data_manager._user_in_group(user_info, "admin") is True
        assert data_manager._user_in_group(user_info, "user") is True
        assert data_manager._user_in_group(user_info, "manager") is False

    def test_user_in_prefixed_group(self, data_manager):
        """Test prefixed group membership (e.g., client:role)"""
        user_info = {"groups": ["frontend-client:admin", "backend:user"], "roles": []}

        assert data_manager._user_in_group(user_info, "admin") is True  # Matches suffix
        assert (
            data_manager._user_in_group(user_info, "frontend-client") is True
        )  # Matches prefix
        assert data_manager._user_in_group(user_info, "user") is True  # Matches suffix
        assert data_manager._user_in_group(user_info, "manager") is False

    def test_get_user_accessible_groups(self, data_manager):
        """Test getting all accessible groups for a user"""
        token = create_test_token("user", ["finance", "marketing"])

        accessible_groups = data_manager.get_user_accessible_groups(token)

        # Should include both groups and roles
        assert "finance" in accessible_groups
        assert "marketing" in accessible_groups
        assert "user" in accessible_groups  # From roles


def create_minimal_token(user_id: str = "testuser", **extra_payload):
    """Create a minimal test token with only required fields"""
    payload = {
        "sub": user_id,
        "iss": "http://test/realms/test",
        "exp": datetime.now(UTC) + timedelta(hours=1),
        **extra_payload,
    }
    return jwt.encode(payload, "secret", algorithm="HS256")


class TestEdgeCasesAndUnsetValues:
    """Test edge cases, unset values, and malformed data"""

    def test_minimal_token_with_no_groups(self, extractor):
        """Test token with minimal required fields and no groups"""
        token = create_minimal_token("minimal_user")

        user_info = extractor.extract_user_info(token)

        assert user_info["user_id"] == "minimal_user"
        assert user_info["groups"] == []
        assert user_info["roles"] == []
        assert user_info["email"] is None
        assert user_info["name"] == " "  # Empty given_name + family_name

    def test_token_with_empty_groups(self, extractor):
        """Test token with explicitly empty groups"""
        token = create_minimal_token("empty_groups", groups=[], roles=[])

        user_info = extractor.extract_user_info(token)

        assert user_info["groups"] == []
        assert user_info["roles"] == []

    def test_token_with_none_values(self, extractor):
        """Test token with None values for optional fields"""
        token = create_minimal_token(
            "none_user", email=None, name=None, groups=None, preferred_username=None
        )

        user_info = extractor.extract_user_info(token)

        assert user_info["user_id"] == "none_user"
        assert user_info["email"] is None
        assert user_info["groups"] == []
        assert user_info["roles"] == []

    def test_token_with_empty_string_values(self, extractor):
        """Test token with empty string values"""
        token = create_minimal_token(
            "empty_user",
            email="",
            name="",
            preferred_username="",
            groups=[""],  # Empty string in groups
        )

        user_info = extractor.extract_user_info(token)

        assert user_info["user_id"] == "empty_user"
        assert user_info["email"] == ""
        assert user_info["groups"] == []  # Empty strings should be filtered out

    def test_token_with_mixed_group_types(self, extractor):
        """Test token with mixed data types in groups"""
        token = create_minimal_token(
            "mixed_user",
            groups=[
                "valid_group",
                "",
                None,
                123,
                {"nested": "dict"},
                ["nested", "list"],
            ],
        )

        user_info = extractor.extract_user_info(token)

        assert "valid_group" in user_info["groups"]
        assert "123" in user_info["groups"]  # Number converted to string
        # Empty strings and None should be filtered out
        assert "" not in user_info["groups"]
        assert None not in user_info["groups"]

    def test_no_user_identifier_raises_error(self, extractor):
        """Test token without any user identifier fields"""
        payload = {
            "iss": "http://test/realms/test",
            "exp": datetime.now(UTC) + timedelta(hours=1),
            # No sub, user_id, uid, preferred_username, or email
        }
        token = jwt.encode(payload, "secret", algorithm="HS256")

        with pytest.raises(ValueError, match="No user identifier found in token"):
            extractor.extract_user_info(token)

    def test_keycloak_complex_structure(self, extractor):
        """Test complex Keycloak token structure with nested roles"""
        token = create_minimal_token(
            "keycloak_user",
            realm_access={
                "roles": [
                    "realm_admin",
                    "realm_user",
                    None,
                    "",
                ]  # Mixed with None/empty
            },
            resource_access={
                "frontend-app": {"roles": ["admin", "user"]},
                "backend-api": {"roles": ["read", "write", ""]},
                "empty-client": {},  # No roles
            },
            groups=["group1", "group2"],
            wids=["windows_role_1", "windows_role_2"],  # Microsoft Graph roles
        )

        user_info = extractor.extract_user_info(token)

        # Check groups include all sources
        assert "group1" in user_info["groups"]
        assert "group2" in user_info["groups"]
        assert "realm_admin" in user_info["groups"]
        assert "realm_user" in user_info["groups"]
        assert "frontend-app:admin" in user_info["groups"]
        assert "frontend-app:user" in user_info["groups"]
        assert "backend-api:read" in user_info["groups"]
        assert "backend-api:write" in user_info["groups"]
        assert "windows_role_1" in user_info["groups"]
        assert "windows_role_2" in user_info["groups"]

        # Check roles
        assert "realm_admin" in user_info["roles"]
        assert "realm_user" in user_info["roles"]

    def test_malformed_realm_access(self, extractor):
        """Test handling of malformed realm_access structure"""
        token = create_minimal_token(
            "malformed_user",
            realm_access="not_a_dict",  # Should be a dict
            resource_access=["not_a_dict"],  # Should be a dict
        )

        # Should not raise error, just ignore malformed data
        user_info = extractor.extract_user_info(token)
        assert user_info["user_id"] == "malformed_user"
        assert user_info["groups"] == []
        assert user_info["roles"] == []


class TestLoggingBehavior:
    """Test logging behavior for various scenarios"""

    def test_logging_when_no_groups_found(self, extractor):
        """Test that appropriate warning logging occurs when no groups are found"""
        with patch.object(extractor, "logger") as mock_logger:
            token = create_minimal_token("no_groups_user")

            user_info = extractor.extract_user_info(token)

            # Check that warning log was called for no groups
            mock_logger.warning.assert_called_with(
                "User no_groups_user has no groups or roles assigned"
            )
            # Check that debug log was called with group count
            mock_logger.debug.assert_called_with(
                "Extracted user info: no_groups_user with 0 groups"
            )
            assert user_info["groups"] == []

    def test_logging_on_token_extraction_error(self, extractor):
        """Test error logging when token extraction fails"""
        with patch.object(extractor, "logger") as mock_logger:
            with pytest.raises(AssertionError):
                extractor.extract_user_info("completely-invalid-token")

            # Verify error was logged
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert "Failed to decode JWT token" in error_call

    def test_data_manager_logging_on_success(self, data_manager):
        """Test data manager success logging"""
        with patch.object(data_manager, "logger") as mock_logger:
            token = create_test_token("log_user", ["test_group"])

            data_manager.resolve_data_resource(token, "test_group")

            # Check info logging
            mock_logger.info.assert_called()
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Resolved group data resource" in call for call in info_calls)

    def test_data_manager_logging_on_error(self, data_manager):
        """Test data manager error logging"""
        with patch.object(data_manager, "logger") as mock_logger:
            with pytest.raises(AssertionError):
                data_manager.resolve_data_resource("invalid-token")

            # Verify error was logged
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert "Failed to resolve data resource" in error_call
        assert "Failed to resolve data resource" in error_call


class TestNormalizeGroups:
    """Test the _normalize_groups method with various input types"""

    def test_normalize_groups_list(self, extractor):
        """Test normalizing list of groups"""
        result = extractor._normalize_groups(["group1", "group2", 123, None, ""])

        assert "group1" in result
        assert "group2" in result
        assert "123" in result
        # None and empty strings should be filtered out
        assert len([g for g in result if not g]) == 0

    def test_normalize_groups_string(self, extractor):
        """Test normalizing single string group"""
        result = extractor._normalize_groups("single_group")

        assert result == ["single_group"]

    def test_normalize_groups_dict(self, extractor):
        """Test normalizing dict structure (like resource_access)"""
        groups_dict = {
            "client1": ["role1", "role2"],
            "client2": "single_role",
            "client3": None,
            "client4": [],
        }

        result = extractor._normalize_groups(groups_dict)

        assert "client1:role1" in result
        assert "client1:role2" in result
        assert "client2:single_role" in result
        # client3 has None value, so should be skipped
        assert not any("client3:" in item for item in result)
        # client4 has empty list, so should be skipped
        assert not any("client4:" in item for item in result)

    def test_normalize_groups_empty_or_none(self, extractor):
        """Test normalizing empty or None values"""
        assert extractor._normalize_groups(None) == []
        assert extractor._normalize_groups([]) == []
        assert extractor._normalize_groups({}) == []
        assert extractor._normalize_groups("") == []


class TestGetUserAccessibleGroups:
    """Test getting user accessible groups with error handling"""

    def test_get_user_accessible_groups_success(self, data_manager):
        """Test successful retrieval of user accessible groups"""
        token = create_test_token("access_user", ["group1", "group2"])

        groups = data_manager.get_user_accessible_groups(token)

        assert "group1" in groups
        assert "group2" in groups
        assert "user" in groups  # From realm roles

    def test_get_user_accessible_groups_error(self, data_manager):
        """Test error handling when getting user groups fails"""
        with patch.object(data_manager, "logger") as mock_logger:
            groups = data_manager.get_user_accessible_groups("invalid-token")

            assert groups == []
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert "Failed to get user groups" in error_call


class TestSanitizeIdentifierEdgeCases:
    """Test identifier sanitization with more edge cases"""

    def test_sanitize_dots_sequence(self, data_manager):
        """Test sanitization of multiple consecutive dots"""
        test_cases = [
            ("group..name", "group_name"),
            ("group...name", "group_name"),
            ("group....name", "group_name"),
            ("..group", "_group"),
            ("group..", "group_"),
            ("...", "_"),
            ("group.name.test", "group.name.test"),  # Single dots preserved
        ]

        for input_val, expected in test_cases:
            result = data_manager._sanitize_identifier(input_val)
            assert (
                result == expected
            ), f"Failed for {input_val}: got {result}, expected {expected}"

    def test_sanitize_unicode_and_special_chars(self, data_manager):
        """Test sanitization with unicode and special characters"""
        test_cases = [
            ("group-üñíçødé", "group-üñíçødé"),  # Unicode allowed by isalnum()
            ("group@#$%^&*()", "group"),  # Special chars removed
            ("group name", "groupname"),  # Spaces removed
            ("group\nname", "groupname"),  # Newlines removed
            ("group\tname", "groupname"),  # Tabs removed
            ("group/path\\danger", "grouppathdanger"),  # Path separators removed
        ]

        for input_val, expected in test_cases:
            result = data_manager._sanitize_identifier(input_val)
            assert (
                result == expected
            ), f"Failed for {input_val}: got {result}, expected {expected}"

    def test_sanitize_empty_input(self, data_manager):
        """Test sanitization with empty or whitespace input"""
        test_cases = [
            ("", ""),
            ("   ", ""),
            ("\n\t", ""),
            ("...", "_"),
        ]

        for input_val, expected in test_cases:
            result = data_manager._sanitize_identifier(input_val)
            assert (
                result == expected
            ), f"Failed for {input_val}: got {result}, expected {expected}"
