from app.utils import mask_token, token_fingerprint


def test_mask_token_removes_the_entire_credential():
    token = "secret-provider-token"

    masked = mask_token(f"credential={token}", token)

    assert masked == "credential=[REDACTED]"
    assert "secret" not in masked


def test_token_fingerprint_does_not_include_token_fragments():
    token = "secret-provider-token"

    fingerprint = token_fingerprint(token)

    assert fingerprint.startswith(f"len={len(token)} sha256=")
    assert token[:6] not in fingerprint
    assert token[-6:] not in fingerprint
