import copy
import app.vars as vars_module

_SCHEMA_DROP_KEYS = {
    "description",
    "examples",
    "example",
    "default",
    "title",
    "$id",
    "$schema",
    "deprecated",
    "readOnly",
    "writeOnly",
}


def prune_schema(schema, depth=2):
    """Trim verbose JSON schema fields to keep payloads compact."""

    if not isinstance(schema, (dict, list)) or schema is None:
        return schema

    if isinstance(schema, list):
        return [prune_schema(item, depth) for item in schema]

    pruned = {}
    for key, value in schema.items():
        if key in _SCHEMA_DROP_KEYS:
            continue
        if key == "properties":
            if depth <= 0:
                pruned[key] = {}
            else:
                pruned[key] = {
                    prop: prune_schema(prop_schema, depth - 1)
                    for prop, prop_schema in value.items()
                }
            continue
        if key == "items":
            if depth <= 0:
                if isinstance(value, dict):
                    pruned[key] = {"type": value.get("type", "object")}
                else:
                    pruned[key] = value
            else:
                pruned[key] = prune_schema(value, depth - 1)
            continue
        if key in {"allOf", "anyOf", "oneOf"}:
            if depth <= 0:
                continue
            pruned[key] = [prune_schema(item, depth - 1) for item in value]
            continue
        if key == "additionalProperties":
            # Preserve booleans, otherwise prune recursively.
            if isinstance(value, bool) or depth <= 0:
                pruned[key] = value
            else:
                pruned[key] = prune_schema(value, depth - 1)
            continue

        pruned[key] = prune_schema(value, depth)

    return pruned


def inline_schema(schema, top_level_schema, seen=None):
    """
    Recursively inlines JSON schema references ($ref).

    Args:
        schema (dict or list or any): The schema object or part of the schema to process.
        top_level_schema (dict): The complete top-level schema document for reference resolution.
        seen (set, optional): A set to track seen references to prevent infinite recursion.
                               Defaults to None, and a new set is created on the initial call.

    Returns:
        dict or list or any: The schema with all references inlined.
    """
    if seen is None:
        seen = set()

    if not isinstance(schema, (dict, list)) or schema is None:
        return schema

    if isinstance(schema, list):
        return [inline_schema(item, top_level_schema, seen) for item in schema]
    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path in seen:
            # Prevent infinite recursion for cyclic refs
            return {}
        seen.add(ref_path)

        try:
            if ref_path.startswith("#/"):
                path_parts = ref_path[2:].split("/")
                definition = top_level_schema
                for part in path_parts:
                    part = part.replace("~1", "/").replace("~0", "~")
                    if isinstance(definition, dict):
                        definition = definition.get(part)
                    elif isinstance(definition, list):
                        try:
                            index = int(part)
                        except ValueError as exc:
                            raise ValueError(
                                f"Schema definition not found for ref: {ref_path}"
                            ) from exc
                        if index < 0 or index >= len(definition):
                            raise ValueError(
                                f"Schema definition not found for ref: {ref_path}"
                            )
                        definition = definition[index]
                    else:
                        raise ValueError(
                            f"Schema definition not found for ref: {ref_path}"
                        )

                if definition is None:
                    raise ValueError(f"Schema definition not found for ref: {ref_path}")

                return inline_schema(definition, top_level_schema, seen)
            else:
                # Handle other types of references, like definitions in $defs
                def_name = ref_path.split("/")[-1]
                definition = top_level_schema.get("$defs", {}).get(def_name)
                if definition is None:
                    raise ValueError(f"Schema definition not found for ref: {ref_path}")
                return inline_schema(definition, top_level_schema, seen)
        finally:
            seen.discard(ref_path)

    new_schema = {}
    for key, value in schema.items():
        new_schema[key] = inline_schema(value, top_level_schema, seen)

    return new_schema


def map_tools(tools, include_output_schema=False):
    """
    Maps a list of tool objects to a new format with inlined schemas.

    Args:
        tools (list): A list of tool objects, where each object has a 'name', 'description',
                      and 'inputSchema'.
        include_output_schema (bool): If True, include the outputSchema in the mapped tools.
                                      This is useful for UI generation where the output structure
                                      needs to be known. Defaults to False for backwards compatibility.

    Returns:
        list: A list of mapped tool objects with inlined schemas.
    """
    mapped_tools = []
    for tool in tools:
        # Support both dict-like and object-like tool definitions
        name = None
        description = None
        input_schema = {}
        output_schema = None

        if isinstance(tool, dict):
            name = tool.get("name") or tool.get("function", {}).get("name")
            description = tool.get("description") or tool.get("function", {}).get(
                "description"
            )
            input_schema = tool.get("inputSchema", {}) or tool.get("function", {}).get(
                "parameters", {}
            )
            output_schema = tool.get("outputSchema") or tool.get("function", {}).get(
                "outputSchema"
            )
            if output_schema is None:
                output_schema = vars_module.get_tool_output_schema(name)
        else:
            # Object-like tool definitions (e.g., mcp Tool, pydantic Tool)
            if hasattr(tool, "name"):
                name = getattr(tool, "name")
            if hasattr(tool, "function"):
                func = getattr(tool, "function")
                name = name or getattr(func, "name", None)
                description = description or getattr(func, "description", None)
                if hasattr(func, "parameters"):
                    input_schema = getattr(func, "parameters") or {}
                if hasattr(func, "outputSchema"):
                    output_schema = getattr(func, "outputSchema")
            description = description or getattr(tool, "description", None)
            input_schema = input_schema or getattr(tool, "inputSchema", {}) or {}
            output_schema = output_schema or getattr(tool, "outputSchema", None)
            if output_schema is None:
                output_schema = vars_module.get_tool_output_schema(name)

        if not name:
            continue

        input_schema_copy = copy.deepcopy(input_schema)
        processed_schema = inline_schema(input_schema_copy, input_schema_copy)
        processed_schema = prune_schema(processed_schema)

        # The top-level $defs is no longer needed after inlining
        if "$defs" in processed_schema:
            del processed_schema["$defs"]

        # Remove any input properties that are mapped to headers
        mapping = getattr(vars_module, "MCP_MAP_HEADER_TO_INPUT", {}) or {}
        if isinstance(processed_schema, dict) and processed_schema.get("properties"):
            props = processed_schema.get("properties", {})
            for input_prop in list(props.keys()):
                if input_prop in mapping:
                    props.pop(input_prop, None)
                    required = processed_schema.get("required")
                    if isinstance(required, list) and input_prop in required:
                        processed_schema["required"] = [
                            r for r in required if r != input_prop
                        ]
            if not processed_schema.get("properties"):
                processed_schema = {}

        if not isinstance(processed_schema, dict) or not processed_schema:
            processed_schema = {"type": "object"}
        elif processed_schema.get("type") is None:
            processed_schema["type"] = "object"

        mapped_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": processed_schema,
            },
        }

        # Optionally include output schema for UI generation
        if include_output_schema and output_schema:
            output_schema_copy = copy.deepcopy(output_schema)
            processed_output = inline_schema(output_schema_copy, output_schema_copy)
            processed_output = prune_schema(
                processed_output, depth=3
            )  # More depth for output

            # Remove top-level $defs after inlining
            if "$defs" in processed_output:
                del processed_output["$defs"]

            mapped_tool["function"]["outputSchema"] = processed_output

        mapped_tools.append(mapped_tool)

    return mapped_tools
