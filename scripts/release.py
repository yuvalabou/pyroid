#!/usr/bin/env python3
"""PyRoid Executable Release Pipeline & Automation Script.

This script automates building, verifying, packaging, and releasing the standalone
PyRoid Windows executable. It can run on-demand, continuously in watch mode on code
updates, or as a Git post-commit hook.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = REPO_ROOT / "pyroid.spec"
DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"
PYPROJECT_FILE = REPO_ROOT / "pyproject.toml"


def get_project_version() -> str:
    """Extract project version from pyproject.toml."""
    if not PYPROJECT_FILE.exists():
        return "1.0.0"

    with open(PYPROJECT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("version ="):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip('"\'')
    return "1.0.0"


def get_git_commit_hash() -> Optional[str]:
    """Get the current short Git commit hash if in a Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def calculate_sha256(filepath: Path) -> str:
    """Calculate the SHA256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_tests() -> bool:
    """Run pytest suite before releasing."""
    print("==> [1/5] Running automated tests with pytest...")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(REPO_ROOT),
    )
    if res.returncode != 0:
        print("[-] Tests failed! Aborting release build to avoid distributing broken binaries.")
        return False
    print("[+] All tests passed successfully.")
    return True


def clean_build_artifacts(dist_dir: Path) -> None:
    """Clean temporary build directories."""
    print("==> [2/5] Cleaning previous build artifacts...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    target_dist_app = dist_dir / "pyroid"
    if target_dist_app.exists():
        shutil.rmtree(target_dist_app, ignore_errors=True)
    print("[+] Clean complete.")


def build_executable() -> bool:
    """Run PyInstaller with the project's spec file."""
    print("==> [3/5] Compiling PyRoid standalone executable via PyInstaller...")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", str(SPEC_FILE)]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print("[-] PyInstaller build failed!")
        return False
    print("[+] Compilation finished successfully.")
    return True


def verify_executable(dist_dir: Path) -> bool:
    """Verify that the built executable runs correctly."""
    print("==> [4/5] Verifying executable binary integrity...")
    exe_path = dist_dir / "pyroid" / ("pyroid.exe" if sys.platform == "win32" else "pyroid")
    if not exe_path.exists():
        print(f"[-] Executable not found at expected location: {exe_path}")
        return False

    # Smoke test CLI help
    try:
        test_run = subprocess.run(
            [str(exe_path), "--cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if test_run.returncode != 0:
            print(f"[-] Executable test run failed with code {test_run.returncode}!")
            print(test_run.stderr)
            return False
        if "PyRoid" not in test_run.stdout:
            print("[-] Executable smoke test output does not contain expected header!")
            return False
    except Exception as e:
        print(f"[-] Verification run encountered exception: {e}")
        return False

    print(f"[+] Executable smoke verification passed: {exe_path}")
    return True


def create_release_package(dist_dir: Path, version: str) -> Optional[Path]:
    """Create a distributable zip archive and release manifest."""
    print("==> [5/5] Packaging release bundle and calculating checksums...")
    app_folder = dist_dir / "pyroid"
    if not app_folder.exists():
        print(f"[-] App directory not found: {app_folder}")
        return None

    # Copy README & LICENSE into the distributable folder
    for doc_name in ["README.md", "LICENSE"]:
        src = REPO_ROOT / doc_name
        if src.exists():
            shutil.copy2(src, app_folder / doc_name)

    # Name zip archive
    zip_basename = f"pyroid-v{version}-windows-x64"
    zip_dest = dist_dir / zip_basename
    archive_path = Path(shutil.make_archive(str(zip_dest), "zip", root_dir=str(dist_dir), base_dir="pyroid"))

    # Compute SHA256
    sha256 = calculate_sha256(archive_path)
    sha_file = archive_path.with_name(f"{archive_path.name}.sha256")
    sha_file.write_text(f"{sha256}  {archive_path.name}\n", encoding="utf-8")

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    git_commit = get_git_commit_hash()

    manifest = {
        "name": "PyRoid",
        "version": version,
        "platform": "windows-x64",
        "git_commit": git_commit,
        "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "release_zip": archive_path.name,
        "release_zip_size_mb": round(size_mb, 2),
        "sha256": sha256,
        "executable_relpath": "pyroid/pyroid.exe",
    }
    manifest_file = dist_dir / "release-manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[+] Release package created: {archive_path}")
    print(f"[+] Size: {size_mb:.2f} MB")
    print(f"[+] SHA256: {sha256}")
    print(f"[+] Manifest: {manifest_file}")
    return archive_path


def build_release(skip_tests: bool = False, dist_dir: Optional[Path] = None) -> bool:
    """Execute full release build pipeline."""
    target_dist = dist_dir or DIST_DIR
    target_dist.mkdir(parents=True, exist_ok=True)
    version = get_project_version()

    print("=" * 65)
    print(f" PyRoid Executable Release Pipeline - v{version}")
    print("=" * 65)

    if not skip_tests:
        if not run_tests():
            return False
    else:
        print("[!] Skipping test execution (--skip-tests active)")

    clean_build_artifacts(target_dist)

    if not build_executable():
        return False

    if not verify_executable(target_dist):
        return False

    archive = create_release_package(target_dist, version)
    if not archive:
        return False

    print("=" * 65)
    print(f" SUCCESS: PyRoid v{version} release build complete!")
    print(f" Output Location: {target_dist}")
    print("=" * 65)
    return True


def install_git_hook() -> bool:
    """Install Git post-commit hook to trigger release builds on commit."""
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        print("[-] .git directory not found. Not a git repository.")
        return False

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    post_commit_hook = hooks_dir / "post-commit"

    script_relpath = Path("scripts") / "release.py"
    hook_content = (
        "#!/bin/sh\n"
        "# PyRoid auto-release hook\n"
        "echo '[PyRoid Git Hook] Code committed. Triggering release pipeline...'\n"
        f"uv run python {script_relpath} --skip-tests\n"
    )

    post_commit_hook.write_text(hook_content, encoding="utf-8")
    try:
        # Make executable on unix-like environments if applicable
        post_commit_hook.chmod(0o755)
    except Exception:
        pass

    print(f"[+] Installed Git post-commit hook at: {post_commit_hook}")
    return True


def get_source_file_mtimes() -> Dict[Path, float]:
    """Collect modification timestamps for all relevant source files."""
    mtimes: Dict[Path, float] = {}

    # Watch source python files and project configs
    watch_dirs = [REPO_ROOT / "pyroid", REPO_ROOT / "tests"]
    watch_files = [REPO_ROOT / "main.py", REPO_ROOT / "pyproject.toml", SPEC_FILE]

    for p in watch_files:
        if p.exists():
            mtimes[p] = p.stat().st_mtime

    for d in watch_dirs:
        if d.exists():
            for p in d.rglob("*.py"):
                mtimes[p] = p.stat().st_mtime

    return mtimes


def watch_and_release(debounce_sec: float = 2.0, skip_tests: bool = False) -> None:
    """Watch codebase for changes and automatically rebuild release on updates."""
    print("=" * 65)
    print(" PyRoid Code Watcher & Auto-Release Mode Active")
    print(f" Watching pyroid/, main.py, pyproject.toml (Debounce: {debounce_sec}s)")
    print(" Press Ctrl+C to stop watching.")
    print("=" * 65)

    last_mtimes = get_source_file_mtimes()

    try:
        while True:
            time.sleep(1.0)
            current_mtimes = get_source_file_mtimes()
            changed_files: List[Path] = []

            for path, mtime in current_mtimes.items():
                if path not in last_mtimes or mtime > last_mtimes[path]:
                    changed_files.append(path)

            if changed_files:
                rel_names = [str(p.relative_to(REPO_ROOT)) for p in changed_files[:3]]
                more_suffix = f" (+{len(changed_files) - 3} more)" if len(changed_files) > 3 else ""
                print(f"\n[Watch Event] Change detected in: {', '.join(rel_names)}{more_suffix}")
                print(f"Debouncing for {debounce_sec} seconds before release build...")
                time.sleep(debounce_sec)

                # Re-snapshot timestamps to prevent immediate re-triggering
                last_mtimes = get_source_file_mtimes()

                # Trigger release build
                try:
                    build_release(skip_tests=skip_tests)
                except Exception as e:
                    print(f"[-] Release build encountered error: {e}")

                print("\n[Watcher] Resuming file watch. Waiting for code updates...")
            else:
                last_mtimes = current_mtimes
    except KeyboardInterrupt:
        print("\n[Watcher] Code watcher stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PyRoid Executable Release Pipeline & Automation Script"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest suite before building",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean build and dist artifacts, then exit",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch codebase for updates and automatically rebuild executable on change",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Debounce duration in seconds for watch mode (default: 2.0)",
    )
    parser.add_argument(
        "--install-git-hook",
        action="store_true",
        help="Install Git post-commit hook to trigger release build on code commits",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for release artifacts",
    )

    args = parser.parse_args()

    if args.install_git_hook:
        success = install_git_hook()
        return 0 if success else 1

    if args.clean_only:
        clean_build_artifacts(Path(args.output_dir) if args.output_dir else DIST_DIR)
        return 0

    if args.watch:
        watch_and_release(debounce_sec=args.debounce, skip_tests=args.skip_tests)
        return 0

    dist_dir = Path(args.output_dir) if args.output_dir else DIST_DIR
    success = build_release(skip_tests=args.skip_tests, dist_dir=dist_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
