---
name: release-executable
description: Builds the standalone Windows executable for PyRoid, verifies binary integrity, and creates a packaged release zip with checksums whenever code is updated or released.
---

# Release Executable Skill (`release-executable`)

Use this skill whenever you need to build, test, verify, package, or automate releases of the standalone PyRoid executable (`pyroid.exe`) on Windows.

## Quick Trigger Commands

- **Full Release Pipeline** (Runs tests, builds PyInstaller exe, tests binary integrity, packages `.zip` + `.sha256` + manifest):
  ```powershell
  uv run python scripts/release.py
  ```
  *(Alternatively on Windows: `.\scripts\build_release.bat`)*

- **Fast Build (Skip Tests)**:
  ```powershell
  uv run python scripts/release.py --skip-tests
  ```

- **Watch Mode (Auto-Rebuild on Code Update)**:
  Runs continuously in the background, watches `pyroid/`, `main.py`, and `pyproject.toml`, debounces changes, and rebuilds the executable upon code update:
  ```powershell
  uv run python scripts/release.py --watch
  ```

- **Git Commit Automation**:
  Install or refresh the Git `post-commit` hook so every `git commit` automatically triggers a release:
  ```powershell
  uv run python scripts/release.py --install-git-hook
  ```

## Release Artifacts Produced

All release artifacts are generated into the `dist/` directory:
- `dist/pyroid/`: Standalone application directory containing `pyroid.exe` and bundled Qt/VTK/PyVista dependencies.
- `dist/pyroid-v<version>-windows-x64.zip`: Compressed release bundle with documentation (`README.md`, `LICENSE`).
- `dist/pyroid-v<version>-windows-x64.zip.sha256`: Cryptographic SHA-256 verification hash.
- `dist/release-manifest.json`: JSON metadata including UTC timestamp, version, git commit hash, file size, and hashes.

## Verification Checklist

1. Run `dist\pyroid\pyroid.exe --cli --help` to ensure CLI arguments parse cleanly.
2. Run `dist\pyroid\pyroid.exe --cli -o test.stl --res 30` to verify 3D marching cubes and mesh generation work in frozen runtime.
3. Launch `dist\pyroid\pyroid.exe` to verify the PyQt5 GUI and 3D PyVista viewport load without missing Qt/VTK shared libraries.
