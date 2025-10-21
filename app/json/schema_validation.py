from jsonschema import validate


def validate_schema(schema: dict, data: dict) -> None:
    """
    Validate data against a JSON schema.
    Wrapper around jsonschema.validate
    """

    validate(instance=data, schema=schema)
