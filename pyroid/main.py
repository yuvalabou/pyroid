#!/usr/bin/env python3
"""Main application entry point dispatcher for PyRoid."""

import sys


def main():
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from pyroid.cli import main as cli_main

        return cli_main()
    from pyroid.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
