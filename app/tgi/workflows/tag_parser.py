"""
Pure functions for extracting, stripping and inspecting XML-style tags
that the LLM embeds in its output (``<reroute>``, ``<user_feedback_needed>``,
``<passthrough>``, ``<return>``, ``<no_reroute>``, etc.).

Every function is stateless — no class required.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedTag:
    """Result of parsing an XML-style tag with optional attributes."""

    content: Optional[str]
    attrs: dict[str, str]


def extract_tag_with_attrs(text: str, tag: str) -> ParsedTag:
    """Extract a ``<tag attr="val">content</tag>`` block, returning content + attrs."""
    match = re.search(
        rf"<{tag}([^>]*)>(?P<content>.*?)</{tag}>",
        text or "",
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ParsedTag(None, {})
    attrs_raw = match.group(1) or ""
    attrs: dict[str, str] = {}
    for attr_match in re.finditer(r'(\w+)\s*=\s*(["\'])(.*?)\2', attrs_raw):
        attrs[attr_match.group(1)] = attr_match.group(3)
    content = match.group("content").strip() if match.group("content") else None
    return ParsedTag(content, attrs)


def extract_tag(text: str, tag: str) -> Optional[str]:
    """Convenience wrapper — returns only the tag content (no attrs)."""
    parsed = extract_tag_with_attrs(text, tag)
    return parsed.content


def extract_run_tag(text: str) -> Optional[bool]:
    """Extract ``<run>true/false</run>`` and return the boolean value."""
    match = re.search(r"<run>(true|false)</run>", text or "", re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return None


def extract_next_agent(text: str) -> Optional[str]:
    """Extract ``<next_agent>name</next_agent>``."""
    next_agent = extract_tag(text, "next_agent")
    if next_agent:
        return next_agent.strip()
    return None


def extract_return_values(text: str) -> list[tuple[str, str]]:
    """
    Extract all ``<return name="...">value</return>`` pairs from LLM output.
    """
    pattern = r"<return\s+name=[\"']([^\"']+)[\"']>(.*?)</return>"
    matches = re.findall(pattern, text or "", re.IGNORECASE | re.DOTALL)
    return [(name.strip(), value.strip()) for name, value in matches if name.strip()]


def strip_tags(text: str) -> str:
    """Remove workflow control tags from text."""
    stripped = re.sub(
        r"<(/?)(reroute|user_feedback_needed|user_feedback|passthrough)([^>]*)>",
        "",
        text or "",
    )
    stripped = re.sub(
        r"<return\b[^>]*>(.*?)</return>",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"<no_reroute>", "", stripped, flags=re.IGNORECASE)


def extract_passthrough_content(text: str) -> str:
    """
    Extract content from ``<passthrough>`` tags for streaming.

    Returns all complete passthrough blocks joined with ``\\n\\n``.
    """
    if not text:
        return ""

    pattern = r"<passthrough>(.*?)</passthrough>"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return ""

    return "\n\n".join(matches) + "\n\n"


def has_no_reroute(content: Optional[str]) -> bool:
    """Check whether the content contains a ``<no_reroute>`` marker."""
    return bool(content and "<no_reroute>" in content)


# ---------------------------------------------------------------------------
# Agent text formatting helpers
# ---------------------------------------------------------------------------


def start_text(agent_def) -> str:
    """Generate a user-friendly status message when an agent starts.

    Note: The agent's description is used as the LLM system prompt,
    not shown to the user. This returns a brief status indicator.
    """
    agent_name_raw = agent_def.agent
    if agent_name_raw.startswith("get_"):
        noun = agent_name_raw.replace("get_", "").replace("_", " ")
        return f"\nFetching your {noun}...\n"
    if agent_name_raw.startswith("ask_"):
        noun = agent_name_raw.replace("ask_", "").replace("_", " ")
        return f"\nAsking for {noun}...\n"
    agent_name = agent_name_raw.replace("_", " ").title()
    return f"\nI will work on the following: {agent_name}...\n"


def append_agent_guidelines(prompt_text: str, agent_def) -> str:
    """Append standard workflow guidelines to an agent system prompt."""
    guidelines = (
        "Workflow guidelines:\n"
        "- If you need more info from the user, respond only with "
        "<user_feedback_needed>Your question</user_feedback_needed>.\n"
        "- If the request does not match the goal or cannot be solved by this workflow, respond only with "
        "<reroute>reason</reroute>.\n"
        "- Respect <no_reroute> if present in the latest user message; otherwise honor reroute signals.\n"
        "- Keep responses concise; include only the necessary tag when using these markers."
    )
    result = f"{prompt_text}\n\n{guidelines}"

    # Add pass-through guideline if specified as a string
    if agent_def.pass_through_guideline:
        result += (
            f"\n\nResponse guideline: {agent_def.pass_through_guideline}\n"
            "Wrap the content you want to show to the user in <passthrough></passthrough> tags. "
            "Only content inside these tags will be visible to the user."
        )

    return result
