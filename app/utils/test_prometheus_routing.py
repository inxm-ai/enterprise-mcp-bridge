from types import SimpleNamespace

from app.utils.prometheus_routing import safe_route_name_resolver


def test_safe_route_name_resolver_supports_two_argument_dependency():
    captured = {}

    def resolver(scope, routes):
        captured["scope"] = scope
        captured["routes"] = routes
        return "two-argument-route"

    safe_resolver = safe_route_name_resolver(resolver)
    scope = {"path": "/tools"}
    result = safe_resolver(
        scope,
        [SimpleNamespace(path="/tools"), SimpleNamespace()],
    )

    assert result == "two-argument-route"
    assert captured["scope"] == scope
    assert [route.path for route in captured["routes"]] == ["/tools"]


def test_safe_route_name_resolver_supports_three_argument_dependency():
    captured = {}

    def resolver(scope, routes, route_name):
        captured["scope"] = scope
        captured["routes"] = routes
        captured["route_name"] = route_name
        return route_name

    safe_resolver = safe_route_name_resolver(resolver)
    scope = {"path": "/tools"}
    result = safe_resolver(
        scope,
        [SimpleNamespace(path="/tools"), SimpleNamespace()],
        "tools",
    )

    assert result == "tools"
    assert captured["scope"] == scope
    assert [route.path for route in captured["routes"]] == ["/tools"]
    assert captured["route_name"] == "tools"
