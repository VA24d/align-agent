"""Claude Code hook entry point: the write-through engine copied beside it by tool/build.sh."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stalecheck_wt.claude_code import main  # noqa: E402

sys.exit(main())
