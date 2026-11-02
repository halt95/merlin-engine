# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Console-script entry for ``merlin``.

The launchers, profiles and model assets live in the Merlin-engine checkout, not in the wheel, so this entry point
finds the checkout (``$MERLIN_HOME``, else this file's directory, else the current directory or a parent) and says so
clearly when there is none.
"""

import os
import sys
from pathlib import Path

_MARK = Path("tools") / "merlin" / "serve_cli.py"


def find_checkout() -> Path | None:
    here = Path(__file__).resolve().parent
    cands = [Path(os.environ["MERLIN_HOME"])] if os.environ.get("MERLIN_HOME") else []
    cands += [here, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for c in cands:
        if (c / _MARK).is_file():
            return c
    return None


def main() -> int:
    root = find_checkout()
    if root is None:
        print(
            "merlin: no Merlin-engine checkout found. Set MERLIN_HOME=<checkout> (it holds tools/merlin, models/ and "
            "the launchers) or run from inside it.",
            file=sys.stderr,
        )
        return 2
    sys.path.insert(0, str(root / "tools" / "merlin"))
    import serve_cli

    serve_cli.R.safe_stdio()
    return serve_cli.main()


if __name__ == "__main__":
    sys.exit(main())
