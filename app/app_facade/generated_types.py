import re
from dataclasses import dataclass
from typing import Sequence

from fastapi import HTTPException


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


@dataclass(frozen=True)
class Scope:
    kind: str  # "group" or "user"
    identifier: str


@dataclass(frozen=True)
class Actor:
    user_id: str
    groups: Sequence[str]

    def is_owner(self, scope: Scope) -> bool:
        if scope.kind == "user":
            return self.user_id == scope.identifier
        return scope.identifier in set(self.groups or [])


def validate_identifier(value: str, field_label: str) -> str:
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_label} must not be empty")
    if not IDENTIFIER_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail=f"{field_label} must match pattern {IDENTIFIER_RE.pattern}",
        )
    return value
