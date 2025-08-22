def mask_token(text: str, token: str) -> str:
    return text.replace(token, f"{token[:4]}****") if token else text
