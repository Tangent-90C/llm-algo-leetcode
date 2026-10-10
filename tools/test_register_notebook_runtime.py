"""Unit checks for the Notebook repository registration helper."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tools.register_notebook_runtime import registered_roots, registration_path, write_registration


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "repository"
        root.mkdir()
        path = registration_path(Path(directory) / "site-packages")
        write_registration(root, path)
        assert path.name == "llm_algo_leetcode_repo.pth"
        assert registered_roots(path) == (root.resolve(),)
    print("notebook runtime registration: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
