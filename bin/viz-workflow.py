#!/usr/bin/env python3
"""
Visualize workflow files using Graphviz.

Usage:
    python bin/viz-workflow.py <workflow.json> [-o output.png]
    python bin/viz-workflow.py tmp/plan_create_or_run.json -o workflow.png
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize workflow JSON files")
    parser.add_argument("workflow", help="Path to workflow JSON file")
    parser.add_argument("-o", "--output", help="Output file (default: workflow.png)")
    parser.add_argument(
        "-f", "--format", default="png", help="Output format (png, svg, pdf, etc.)"
    )
    return parser.parse_args()


def sanitize_label(text: str, max_len: int = 50) -> str:
    """Sanitize and truncate text for use in Graphviz labels."""
    if len(text) <= max_len:
        return text.replace('"', '\\"').replace("\n", "\\n")
    return text[: max_len - 3].replace('"', '\\"').replace("\n", "\\n") + "..."


def get_tools_label(tools: Any) -> str:
    """Extract tools label from various tool format configurations."""
    if not tools:
        return ""

    tool_names = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, str):
                tool_names.append(tool)
            elif isinstance(tool, dict):
                # Advanced tool config: {"tool_name": {...}}
                tool_names.extend(tool.keys())

    if tool_names:
        return (
            "\\n[Tools: "
            + ", ".join(tool_names[:3])
            + ("...]" if len(tool_names) > 3 else "]")
        )
    return ""


def build_graph(workflow: Dict[str, Any]) -> str:
    """Build Graphviz DOT representation of the workflow."""
    flow_id = workflow.get("flow_id", "workflow")
    root_intent = workflow.get("root_intent", "")
    description = workflow.get("description", "")
    agents = workflow.get("agents", [])

    # Start DOT graph
    lines = [
        "digraph workflow {",
        "  rankdir=TB;",
        '  node [shape=box, style=rounded, fontname="DejaVu Sans"];',
        '  edge [fontname="DejaVu Sans", fontsize=10];',
        "",
        "  // Root/Orchestrator node",
    ]

    # Add root orchestrator node
    root_label = f"[Orchestrator]\\n{root_intent}"
    if description:
        root_label += f"\\n{sanitize_label(description, 40)}"

    lines.append(
        f'  root [label="{root_label}", shape=doubleoctagon, style=filled, fillcolor=lightblue];'
    )
    lines.append("")

    # Track all agent nodes
    agent_names = {agent["agent"] for agent in agents}

    # Add agent nodes
    lines.append("  // Agent nodes")
    for agent in agents:
        agent_name = agent["agent"]
        desc = agent.get("description", "")
        tools = agent.get("tools", [])
        has_when = "when" in agent
        pass_through = agent.get("pass_through", False)

        # Build label
        label = f"{agent_name}"
        if desc:
            label += f"\\n{sanitize_label(desc, 45)}"

        tools_label = get_tools_label(tools)
        if tools_label:
            label += tools_label

        # Styling based on properties
        style = "filled"
        fillcolor = "white"

        if has_when:
            fillcolor = "lightyellow"  # Conditional execution
        if pass_through:
            style = "filled,bold"  # Visible to user
        if not agent.get("depends_on") and not agent.get("tools"):
            fillcolor = "lightgray"  # Terminal/summary nodes

        lines.append(
            f'  {agent_name} [label="{label}", style="{style}", fillcolor="{fillcolor}"];'
        )

    lines.append("")

    # Add edges from root to first agents (no dependencies)
    lines.append("  // Root to entry points")
    first_agents = [agent["agent"] for agent in agents if not agent.get("depends_on")]
    for agent_name in first_agents:
        lines.append(f'  root -> {agent_name} [label="start", color=blue, penwidth=2];')

    lines.append("")

    # Add dependency edges
    lines.append("  // Dependencies")
    for agent in agents:
        agent_name = agent["agent"]
        depends_on = agent.get("depends_on", [])

        for dep in depends_on:
            # Check if dependency exists
            if dep in agent_names:
                label = "depends"
                lines.append(f'  {dep} -> {agent_name} [label="{label}", style=solid];')
            else:
                # Broken dependency - show in red
                lines.append(
                    f'  {dep} -> {agent_name} [label="missing!", color=red, style=dashed];'
                )

    lines.append("")

    # Add reroute edges
    lines.append("  // Reroutes")
    for agent in agents:
        agent_name = agent["agent"]
        reroutes = agent.get("reroute", [])
        on_tool_error = agent.get("on_tool_error")

        if isinstance(reroutes, list):
            for reroute in reroutes:
                on_codes = reroute.get("on", [])
                to_agent = reroute.get("to", "")

                if to_agent:
                    codes_label = ", ".join(on_codes[:2]) + (
                        "..." if len(on_codes) > 2 else ""
                    )
                    lines.append(
                        f"  {agent_name} -> {to_agent} "
                        f'[label="{codes_label}", color=orange, style=dashed, constraint=false];'
                    )

        # Add on_tool_error edge
        if on_tool_error:
            lines.append(
                f"  {agent_name} -> {on_tool_error} "
                f'[label="on error", color=red, style=dashed, constraint=false];'
            )

    lines.append("")

    # Add legend
    lines.extend(
        [
            "  // Legend",
            "  subgraph cluster_legend {",
            '    label="Legend";',
            "    style=dashed;",
            "    fontsize=10;",
            "    ",
            '    legend_normal [label="Normal Agent", style=filled, fillcolor=white];',
            '    legend_conditional [label="Conditional", style=filled, fillcolor=lightyellow];',
            '    legend_terminal [label="Terminal/Summary", style=filled, fillcolor=lightgray];',
            '    legend_visible [label="Visible Output", style="filled,bold", fillcolor=white];',
            "    ",
            '    legend_normal -> legend_conditional [label="dependency", style=invisible];',
            "  }",
        ]
    )

    lines.append("}")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Check if graphviz is available
    try:
        import graphviz
    except ImportError:
        print(
            "Error: graphviz package not found. Install with: pip install graphviz",
            file=sys.stderr,
        )
        print(
            "Also ensure graphviz system package is installed (e.g., apt install graphviz)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load workflow
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {workflow_path}", file=sys.stderr)
        sys.exit(1)

    with open(workflow_path) as f:
        workflow = json.load(f)

    # Generate DOT graph
    dot_source = build_graph(workflow)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = workflow_path.with_suffix(f".{args.format}")

    # Render using graphviz
    try:
        graph = graphviz.Source(dot_source)
        output_file = str(output_path.with_suffix(""))  # graphviz adds extension
        graph.render(output_file, format=args.format, cleanup=True)
        print(f"✅ Workflow visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error rendering graph: {e}", file=sys.stderr)
        print("\nGenerated DOT source:")
        print(dot_source)
        sys.exit(1)


if __name__ == "__main__":
    main()
