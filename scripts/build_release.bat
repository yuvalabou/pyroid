@echo off
rem PyRoid Quick Release Builder for Windows
cd /d "%~dp0\.."
echo ========================================================
echo Running PyRoid Executable Release Pipeline
echo ========================================================
uv run python scripts\release.py %*
