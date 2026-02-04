#!/usr/bin/env python3
"""
Visualize workflow files using Graphviz.

Supports multi-workflow visualization by following references in reroutes.

Usage:
    python bin/viz-workflow.py <workflow.json> [-o output.png]
    python bin/viz-workflow.py tmp/plan_create_or_run.json -o workflow.png

Workflow references in reroutes (e.g., workflows[plan_run]) will automatically
load referenced workflow files from the same directory.
"""

import json
import sys
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional


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


def extract_workflow_references(workflow: Dict[str, Any], base_dir: Path) -> Set[str]:
    """
    Extract workflow references from reroutes and other configurations.

    Looks for patterns like workflows[workflow_name] in all nested structures,
    including in ask.expected_responses.

    Args:
        workflow: The workflow definition.
        base_dir: The base directory for resolving relative paths.

    Returns:
        Set of workflow names referenced.
    """
    references = set()
    workflow_pattern = re.compile(r"workflows\s*\[\s*([^\]]+)\s*\]")

    def search_value(obj):
        """Recursively search through data structures."""
        if isinstance(obj, str):
            matches = workflow_pattern.findall(obj)
            references.update(matches)
        elif isinstance(obj, dict):
            for value in obj.values():
                search_value(value)
        elif isinstance(obj, list):
            for item in obj:
                search_value(item)

    search_value(workflow)
    return references


def load_workflow(workflow_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a workflow JSON file.

    Args:
        workflow_path: Path to the workflow JSON file.

    Returns:
        Parsed workflow dict, or None if file doesn't exist or is invalid.
    """
    if not workflow_path.exists():
        return None

    try:
        with open(workflow_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_all_workflows(
    initial_workflow_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Path]:
    """
    Recursively load a workflow and all referenced workflows.

    Args:
        initial_workflow_path: Path to the initial workflow JSON file.

    Returns:
        Tuple of (workflows_dict, base_dir) where workflows_dict maps
        flow_id to workflow definition.
    """
    base_dir = initial_workflow_path.parent
    workflows = {}
    to_process = {initial_workflow_path.stem}
    processed = set()

    while to_process:
        workflow_name = to_process.pop()
        if workflow_name in processed:
            continue
        processed.add(workflow_name)

        workflow_path = base_dir / f"{workflow_name}.json"
        workflow = load_workflow(workflow_path)

        if workflow is None:
            # Silently skip missing files
            continue

        flow_id = workflow.get("flow_id", workflow_name)
        workflows[flow_id] = workflow

        # Find references to other workflows
        refs = extract_workflow_references(workflow, base_dir)
        for ref in refs:
            if ref not in processed:
                to_process.add(ref)

    return workflows, base_dir


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
        # Show max 2 tools, keep it compact
        shown = ", ".join(tool_names[:2])
        if len(tool_names) > 2:
            shown += f" +{len(tool_names)-2}"
        return f"\\n[{shown}]"
    return ""


def build_subgraph_for_workflow(
    workflow: Dict[str, Any],
    all_workflows: Dict[str, Dict[str, Any]],
    workflow_pattern: re.Pattern,
) -> Tuple[List[str], List[str], Set[Tuple[str, str, str]]]:
    """
    Build graph nodes and edges for a single workflow.

    Returns:
        Tuple of (node_lines, edge_lines, cross_workflow_edges)
        where cross_workflow_edges are (from_agent, to_workflow, to_node_label)
    """
    flow_id = workflow.get("flow_id", "workflow")
    root_intent = workflow.get("root_intent", "")
    description = workflow.get("description", "")
    agents = workflow.get("agents", [])

    node_lines = []
    edge_lines = []
    cross_workflow_edges = set()

    # Create workflow prefix for node naming
    prefix = f"{flow_id}_"

    # Add subgraph for this workflow
    root_label = f"[Orchestrator: {flow_id}]\\n{root_intent}"
    if description:
        root_label += f"\\n{sanitize_label(description, 40)}"

    node_lines.append(f"  subgraph cluster_{flow_id} {{")
    node_lines.append(f'    label="{sanitize_label(root_intent, 35)}";')
    node_lines.append(f"    style=filled;")
    node_lines.append(f"    fillcolor=lightblue;")
    node_lines.append(f"    color=blue;")
    node_lines.append(f"    penwidth=1.5;")
    node_lines.append(f"    margin=0.1;")
    node_lines.append(f"    fontsize=9;")
    node_lines.append("")

    # Track all agent nodes in this workflow
    agent_names = {agent["agent"] for agent in agents}

    # Add agent nodes
    for agent in agents:
        agent_name = agent["agent"]
        agent_id = f"{prefix}{agent_name}"
        desc = agent.get("description", "")
        tools = agent.get("tools", [])
        has_when = "when" in agent
        pass_through = agent.get("pass_through", False)

        # Build label - more compact
        label = f"{agent_name}"
        if desc and len(desc) > 0:
            # Only add first sentence or truncated description
            first_sent = desc.split(".")[0]
            label += f"\\n{sanitize_label(first_sent, 30)}"

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

        node_lines.append(
            f'    {agent_id} [label="{label}", style="{style}", fillcolor="{fillcolor}", fontsize=8, height=0.3, margin="0.05,0.05"];'
        )

    # Add root to first agents (no dependencies)
    root_id = f"{prefix}root"
    node_lines.append(
        f'    {root_id} [label="[Start]", shape=circle, style=filled, fillcolor=yellow];'
    )
    node_lines.append("")

    first_agents = [agent["agent"] for agent in agents if not agent.get("depends_on")]
    for agent_name in first_agents:
        agent_id = f"{prefix}{agent_name}"
        edge_lines.append(
            f'  {root_id} -> {agent_id} [label="start", color=blue, penwidth=2];'
        )

    # Add dependency edges within workflow
    for agent in agents:
        agent_name = agent["agent"]
        agent_id = f"{prefix}{agent_name}"
        depends_on = agent.get("depends_on", [])

        for dep in depends_on:
            if dep in agent_names:
                dep_id = f"{prefix}{dep}"
                edge_lines.append(
                    f'  {dep_id} -> {agent_id} [label="depends", style=solid];'
                )
            else:
                # Broken dependency - show in red
                edge_lines.append(
                    f'  missing_{dep} -> {agent_id} [label="missing!", color=red, style=dashed];'
                )

    # Add reroute edges (including workflow transitions)
    for agent in agents:
        agent_name = agent["agent"]
        agent_id = f"{prefix}{agent_name}"
        reroutes = agent.get("reroute", [])
        on_tool_error = agent.get("on_tool_error")

        if isinstance(reroutes, list):
            for reroute in reroutes:
                on_codes = reroute.get("on", [])
                to_target = reroute.get("to", "")

                # First check direct to target
                if to_target:
                    # Check if this is a workflow reference
                    match = workflow_pattern.match(to_target)
                    if match:
                        target_workflow = match.group(1)
                        cross_workflow_edges.add((agent_id, target_workflow, to_target))
                        codes_label = ", ".join(on_codes[:2]) + (
                            "..." if len(on_codes) > 2 else ""
                        )
                        edge_lines.append(
                            f"  {agent_id} -> {target_workflow}_root "
                            f'[label="{codes_label} → {target_workflow}", color=purple, style=dashed, constraint=false, penwidth=2];'
                        )
                    else:
                        # Regular agent reroute within workflow
                        codes_label = ", ".join(on_codes[:2]) + (
                            "..." if len(on_codes) > 2 else ""
                        )
                        if to_target in agent_names:
                            to_id = f"{prefix}{to_target}"
                            edge_lines.append(
                                f"  {agent_id} -> {to_id} "
                                f'[label="{codes_label}", color=orange, style=dashed, constraint=false];'
                            )

                # Check for nested ask.expected_responses structure
                ask_config = reroute.get("ask", {})
                if isinstance(ask_config, dict):
                    expected_responses = ask_config.get("expected_responses", [])
                    if isinstance(expected_responses, list):
                        for response_group in expected_responses:
                            if isinstance(response_group, dict):
                                # Each key in response_group is a response option
                                for (
                                    response_key,
                                    response_value,
                                ) in response_group.items():
                                    if isinstance(response_value, dict):
                                        response_to = response_value.get("to", "")
                                        if response_to:
                                            # Check if this is a workflow reference
                                            match = workflow_pattern.match(response_to)
                                            if match:
                                                target_workflow = match.group(1)
                                                cross_workflow_edges.add(
                                                    (
                                                        agent_id,
                                                        target_workflow,
                                                        response_to,
                                                    )
                                                )
                                                edge_lines.append(
                                                    f"  {agent_id} -> {target_workflow}_root "
                                                    f'[label="{response_key} → {target_workflow}", color=purple, style=dashed, constraint=false, penwidth=2];'
                                                )
                                            else:
                                                # Regular agent response
                                                if response_to in agent_names:
                                                    to_id = f"{prefix}{response_to}"
                                                    edge_lines.append(
                                                        f"  {agent_id} -> {to_id} "
                                                        f'[label="{response_key}", color=orange, style=dashed, constraint=false];'
                                                    )

        # Add on_tool_error edge
        if on_tool_error:
            # Check if error target is a workflow
            match = workflow_pattern.match(on_tool_error)
            if match:
                target_workflow = match.group(1)
                cross_workflow_edges.add((agent_id, target_workflow, on_tool_error))
                edge_lines.append(
                    f"  {agent_id} -> {target_workflow}_root "
                    f'[label="on error → {target_workflow}", color=red, style=dashed, constraint=false, penwidth=2];'
                )
            elif on_tool_error in agent_names:
                to_id = f"{prefix}{on_tool_error}"
                edge_lines.append(
                    f"  {agent_id} -> {to_id} "
                    f'[label="on error", color=red, style=dashed, constraint=false];'
                )

    node_lines.append("  }")
    node_lines.append("")

    return node_lines, edge_lines, cross_workflow_edges


def build_graph(
    workflows: Dict[str, Dict[str, Any]],
) -> str:
    """Build Graphviz DOT representation of all workflows."""

    # Start DOT graph with better layout settings for multiple workflows
    lines = [
        "digraph workflow {",
        "  rankdir=LR;",
        "  compound=true;",
        "  nodesep=0.3;",
        "  ranksep=0.5;",
        '  node [shape=box, style=rounded, fontname="DejaVu Sans", fontsize=9];',
        '  edge [fontname="DejaVu Sans", fontsize=8];',
        "",
    ]

    workflow_pattern = re.compile(r"^workflows\s*\[\s*([^\]]+)\s*\]$")
    all_node_lines = []
    all_edge_lines = []

    # Build subgraphs for each workflow
    for flow_id, workflow in workflows.items():
        node_lines, edge_lines, _ = build_subgraph_for_workflow(
            workflow, workflows, workflow_pattern
        )
        all_node_lines.extend(node_lines)
        all_edge_lines.extend(edge_lines)

    # Add all nodes
    lines.extend(all_node_lines)
    lines.append("")

    # Add all edges
    lines.append("  // All edges")
    lines.extend(all_edge_lines)

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
            '    legend_workflow [label="Workflow Transition", color=purple, style=dashed];',
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

    # Load workflow and all referenced workflows
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {workflow_path}", file=sys.stderr)
        sys.exit(1)

    try:
        workflows, base_dir = load_all_workflows(workflow_path)

        if not workflows:
            print(
                f"Error: Could not load any workflows from {workflow_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Report which workflows were loaded
        if len(workflows) > 1:
            print(
                f"📦 Loaded {len(workflows)} workflow(s): {', '.join(workflows.keys())}",
                file=sys.stderr,
            )

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in workflow file: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate DOT graph
    try:
        dot_source = build_graph(workflows)
    except Exception as e:
        print(f"Error building graph: {e}", file=sys.stderr)
        print("\nWorkflows loaded:")
        for flow_id in workflows:
            print(f"  - {flow_id}")
        sys.exit(1)

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
