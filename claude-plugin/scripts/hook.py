"""Claude Code hook entry point: the write-through engine copied beside it by tool/build.sh."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from align_wt.claude_code import main  # noqa: E402
except Exception as e:  # noqa: BLE001  never block the agent: report and let the tool call proceed
    print(f"ALIGN: disabled ({type(e).__name__}: {e})", file=sys.stderr)
    sys.exit(0)

sys.exit(main())
