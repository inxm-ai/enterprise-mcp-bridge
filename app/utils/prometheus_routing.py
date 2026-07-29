"""Compatibility helpers for Prometheus route-name resolution."""

from collections.abc import Callable, Sequence
from typing import Any


def safe_route_name_resolver(resolver: Callable[..., Any]) -> Callable[..., Any]:
    """Filter pathless FastAPI routes while preserving the resolver signature."""

    def resolve(
        scope: Any,
        routes: Sequence[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        routes_with_paths = tuple(route for route in routes if hasattr(route, "path"))
        return resolver(scope, routes_with_paths, *args, **kwargs)

    return resolve
