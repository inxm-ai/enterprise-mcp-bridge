"""
OAuth token analysis and user/group information extraction
"""

import logging
import jwt
from typing import Dict, Any, List, Optional
import re
import os

logger = logging.getLogger("uvicorn.error")


class UserInfoExtractor:
    """Extract user and group information from OAuth tokens"""

    def __init__(self):
        self.logger = logger
        # Configure group claim names based on OAuth provider
        self.group_claim_names = [
            "groups",  # Common in many OAuth providers
            "roles",  # Keycloak and others
            "realm_access.roles",  # Keycloak realm roles
            "resource_access",  # Keycloak client roles
        ]

    def extract_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Extract user information from OAuth access token
        Returns user ID, email, groups, and other relevant info
        """
        try:
            # Decode JWT without signature verification for info extraction
            payload = jwt.decode(
                access_token,
                options={"verify_signature": False},
                algorithms=["RS256", "HS256"],
            )

            user_info = {
                "user_id": self._extract_user_id(payload),
                "email": payload.get("email") or payload.get("preferred_username"),
                "name": payload.get("name")
                or payload.get("given_name", "") + " " + payload.get("family_name", ""),
                "groups": self._extract_groups(payload),
                "roles": self._extract_roles(payload),
                "sub": payload.get("sub"),
                "iss": payload.get("iss"),
                "exp": payload.get("exp"),
            }

            self.logger.debug(
                f"Extracted user info: {user_info['user_id']} with {len(user_info['groups'])} groups"
            )

            return user_info
        
        except jwt.exceptions.DecodeError:
            self.logger.error("Failed to decode JWT token")
            raise AssertionError("Invalid token format")
        
        except AssertionError as e:
            self.logger.error(f"Token assertion error: {str(e)}")
            raise e

        except Exception as e:
            self.logger.error(f"Failed to extract user info from token: {str(e)}")
            raise ValueError(f"Invalid or malformed access token: {str(e)}")

    def _extract_user_id(self, payload: Dict[str, Any]) -> str:
        """Extract user ID from token payload"""
        # Try different common user ID fields
        for field in ["sub", "user_id", "uid", "preferred_username", "email"]:
            if field in payload and payload[field]:
                return str(payload[field])
        raise ValueError("No user identifier found in token")

    def _extract_groups(self, payload: Dict[str, Any]) -> List[str]:
        """Extract group memberships from token payload"""
        groups = []

        # Check different group claim formats
        if "groups" in payload:
            groups.extend(self._normalize_groups(payload["groups"]))

        # Keycloak realm roles
        if "realm_access" in payload:
            try:
                if (
                    isinstance(payload["realm_access"], dict)
                    and "roles" in payload["realm_access"]
                ):
                    groups.extend(
                        self._normalize_groups(payload["realm_access"]["roles"])
                    )
            except (AttributeError, TypeError):
                # Ignore malformed realm_access structure
                pass

        # Keycloak resource access (client roles)
        if "resource_access" in payload:
            try:
                if isinstance(payload["resource_access"], dict):
                    for client, access in payload["resource_access"].items():
                        if isinstance(access, dict) and "roles" in access:
                            client_roles = [
                                f"{client}:{role}" for role in access["roles"]
                            ]
                            groups.extend(client_roles)
            except (AttributeError, TypeError):
                # Ignore malformed resource_access structure
                pass
                pass

        # Microsoft Graph groups (if available)
        if "wids" in payload:  # Windows Identity roles
            groups.extend(self._normalize_groups(payload["wids"]))

        final_groups = list(set(groups))  # Remove duplicates

        # Log warning if user has no groups at all
        if not final_groups:
            user_id = (
                payload.get("sub") or payload.get("preferred_username") or "unknown"
            )
            self.logger.warning(f"User {user_id} has no groups or roles assigned")

        return final_groups

    def _extract_roles(self, payload: Dict[str, Any]) -> List[str]:
        """Extract roles from token payload"""
        roles = []

        if "roles" in payload:
            roles.extend(self._normalize_groups(payload["roles"]))

        # Keycloak realm roles
        if "realm_access" in payload and "roles" in payload["realm_access"]:
            roles.extend(self._normalize_groups(payload["realm_access"]["roles"]))

        return list(set(roles))

    def _normalize_groups(self, groups_data: Any) -> List[str]:
        """Normalize group data to list of strings"""
        if groups_data is None:
            return []

        if isinstance(groups_data, list):
            # Filter out None, empty strings, and convert to strings
            return [str(g) for g in groups_data if g is not None and str(g).strip()]
        elif isinstance(groups_data, str):
            # Only return non-empty strings
            return [groups_data] if groups_data.strip() else []
        elif isinstance(groups_data, dict):
            # Handle nested structures
            result = []
            for key, value in groups_data.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    # Filter out empty/None values in nested lists
                    valid_values = [
                        v for v in value if v is not None and str(v).strip()
                    ]
                    result.extend([f"{key}:{v}" for v in valid_values])
                elif str(value).strip():  # Only non-empty values
                    result.append(f"{key}:{value}")
            return result
        else:
            # Handle other types by converting to string if not empty
            str_value = str(groups_data).strip()
            return [str_value] if str_value else []


class DataAccessManager:
    """Manage data access based on user groups and permissions"""

    def __init__(self):
        self.user_extractor = UserInfoExtractor()
        self.logger = logger
        # Configurable data resource templates - these could be paths, database tables, etc.
        self.data_resource_templates = {
            "group": os.getenv("MCP_GROUP_DATA_ACCESS_TEMPLATE", "g/{group_id}"),
            "user": os.getenv("MCP_USER_DATA_ACCESS_TEMPLATE", "u/{user_id}"),
            "shared": os.getenv(
                "MCP_SHARED_DATA_ACCESS_TEMPLATE", "shared/{resource_id}"
            ),
        }

    def resolve_data_resource(
        self, access_token: str, requested_group: Optional[str] = None
    ) -> str:
        """
        Resolve data resource identifier based on user token and requested access

        Args:
            access_token: OAuth access token
            requested_group: Optional group name to access group-specific data

        Returns:
            Resolved resource identifier for MCP server (could be path, table name, etc.)

        Raises:
            PermissionError: If user doesn't have access to requested resource
            ValueError: If invalid parameters provided
        """
        try:
            user_info = self.user_extractor.extract_user_info(access_token)

            if requested_group:
                # Check if user is member of requested group
                if not self._user_in_group(user_info, requested_group):
                    raise PermissionError(
                        f"User {user_info['user_id']} is not a member of group '{requested_group}'"
                    )

                # Return group-specific data resource
                data_resource = self.data_resource_templates["group"].format(
                    group_id=self._sanitize_identifier(requested_group)
                )
                self.logger.info(
                    f"Resolved group data resource for user {user_info['user_id']}: {data_resource}"
                )
                return data_resource
            else:
                # Return user-specific data resource
                data_resource = self.data_resource_templates["user"].format(
                    user_id=self._sanitize_identifier(user_info["user_id"])
                )
                self.logger.info(f"Resolved user data resource: {data_resource}")
                return data_resource

        except Exception as e:
            self.logger.error(f"Failed to resolve data resource: {str(e)}")
            raise

    def _user_in_group(self, user_info: Dict[str, Any], group_name: str) -> bool:
        """Check if user is member of specified group"""
        user_groups = user_info.get("groups", [])
        user_roles = user_info.get("roles", [])

        # Check direct group membership
        if group_name in user_groups:
            return True

        # Check role-based access
        if group_name in user_roles:
            return True

        # Check prefixed groups (e.g., "frontend-client:admin")
        for group in user_groups + user_roles:
            if group.endswith(f":{group_name}") or group.startswith(f"{group_name}:"):
                return True

        return False

    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitize identifier to ensure it's safe for use in templates.

        - Allow alphanumeric, hyphens, underscores, dots, and colons
        - Remove path separators and other dangerous characters
        - Collapse any sequences of two or more dots to a single underscore to avoid ".."
        - Enforce a reasonable length limit
        """
        # Filter allowed characters (note: '/' and '\\' are excluded)
        filtered = "".join(c for c in identifier if c.isalnum() or c in "._:-")
        # Replace any occurrences of two or more consecutive dots with an underscore
        sanitized = re.sub(r"\.{2,}", "_", filtered)
        return sanitized[:100]

    def get_user_accessible_groups(self, access_token: str) -> List[str]:
        """Get list of groups user has access to"""
        try:
            user_info = self.user_extractor.extract_user_info(access_token)
            return user_info.get("groups", []) + user_info.get("roles", [])
        except Exception as e:
            self.logger.error(f"Failed to get user groups: {str(e)}")
            return []


def get_data_access_manager() -> DataAccessManager:
    """Factory function to get DataAccessManager instance"""
    return DataAccessManager()
