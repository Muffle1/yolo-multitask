from pathlib import Path
import json
from typing import Any

def load_label_format(config_path: Path | str) -> list[str] | None:
	"""Read label_format list from the config JSON if present."""
	p = Path(config_path)
	if not p.exists():
		return None
	with open(p, "r", encoding="utf-8") as f:
		data = json.load(f)
	return data.get("label_format")

def parse_label_file(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def parse_label_line(line: str, label_format: list[str]) -> dict[str, Any]:
    tokens = line.split()
    if len(tokens) != len(label_format):
        return {"raw": line}

    entry: dict[str, Any] = {}
    for name, token in zip(label_format, tokens):
        if name.startswith("x") or name.startswith("y"):
            try:
                entry[name] = float(token)
            except ValueError:
                entry[name] = token
        else:
            entry[name] = token
    return entry