import os

from app.json.schema_validation import validate_schema


def get_tool_dry_run_response(tool: dict, tool_input: dict) -> dict:
    if not os.environ.get("TGI_URL", None):
        raise ValueError(
            "TGI_URL environment variable is not set, cannot create dry run response."
        )

    input_schema = tool.get("inputSchema", None)
    if input_schema:
        validate_schema(input_schema, tool_input)
    output_schema = tool.get("outputSchema", None)
    pass
