"""Load KEY=VALUE lines from the project .env into os.environ (never prints values)."""
import os
from pathlib import Path


def load_env(path: str | Path | None = None) -> list[str]:
    p = Path(path) if path else Path(__file__).resolve().parents[2] / ".env"
    keys = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ[k.strip()] = v
            keys.append(k.strip())
    return keys
