import re

def to_snake_case(text: str) -> str:
    """Convert arbitrary text to a safe lowercase, snake_case string."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text
