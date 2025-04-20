# /// script
# dependencies = [
#   "toml>=0.10.2"
# ]
# ///
import subprocess
from typing import Literal

import toml


def bump_version(part: Literal["major", "minor", "patch"] = "patch") -> None:
    """Bump version in pyproject.toml file.

    Args:
        part (Literal["major", "minor", "patch"], optional): Version part to increment. Defaults to "patch".

    Raises:
        ValueError: If part is not 'major', 'minor', or 'patch'.
    """
    file_path = "pyproject.toml"

    with open(file_path, "r") as f:
        pyproject = toml.load(f)

    version = pyproject["project"]["version"]
    major, minor, patch = map(int, version.split("."))

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError("Invalid part value. Choose 'major', 'minor', or 'patch'.")

    new_version = f"{major}.{minor}.{patch}"
    subprocess.run(
        [
            "uvx",
            "--from=toml-cli",
            "toml",
            "set",
            "--toml-path=pyproject.toml",
            "project.version",
            new_version,
        ]
    )

    print(f"Version bumped to {major}.{minor}.{patch}")


if __name__ == "__main__":
    bump_version()
