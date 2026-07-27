"""Recursive regex search over a directory tree, run as its own process.

Not a function agent_tools calls directly, and deliberately so: the pattern
comes from a model, and a catastrophically backtracking regular expression
cannot be interrupted from inside Python -- re.search holds the GIL for the
whole match, so a worker thread would freeze the event loop just as thoroughly
as running it inline would. A separate process can simply be killed, which is
what agent_tools does when this overruns.

Standard library only and no project imports, so starting one costs an
interpreter and nothing else.
"""

import fnmatch
import os
import re
import sys
import time

# Directories never worth walking into during a search.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}
MAX_MATCHES = 200
MAX_FILES = 5000


def search(root: str, regex, file_glob: str, deadline: float) -> str:
    """'path:line: text' per hit, newest concern first: stop early. The
    deadline is checked per file and every so often per line, so an ordinary
    slow scan (a huge tree, a broad pattern) ends itself with a note rather
    than waiting to be killed."""
    hits = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if file_glob and not fnmatch.fnmatch(name, file_glob):
                continue
            if time.monotonic() > deadline:
                hits.append("...[stopped: out of time -- narrow the path or the file glob]")
                return "\n".join(hits)
            scanned += 1
            if scanned > MAX_FILES:
                hits.append(f"...[stopped after scanning {MAX_FILES} files]")
                return "\n".join(hits)
            full = os.path.join(dirpath, name)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            hits.append(f"{os.path.relpath(full, root)}:{lineno}: {line.rstrip()}")
                            if len(hits) >= MAX_MATCHES:
                                hits.append(f"...[stopped at {MAX_MATCHES} matches]")
                                return "\n".join(hits)
                        if lineno % 500 == 0 and time.monotonic() > deadline:
                            hits.append("...[stopped: out of time -- narrow the path or the file glob]")
                            return "\n".join(hits)
            except (OSError, UnicodeDecodeError):
                continue  # binary or unreadable -- not what this tool is for
    return "\n".join(hits)


def main(argv: list) -> int:
    if len(argv) != 4:
        sys.stderr.write("usage: file_search.py ROOT PATTERN FILE_GLOB TIMEOUT_SECONDS\n")
        return 2
    root, pattern, file_glob, timeout = argv
    try:
        regex = re.compile(pattern)
    except re.error as e:
        sys.stdout.write(f"[error] '{pattern}' is not a valid regular expression: {e}")
        return 0
    try:
        deadline = time.monotonic() + float(timeout)
    except ValueError:
        sys.stdout.write(f"[error] '{timeout}' is not a number of seconds")
        return 0
    sys.stdout.write(search(root, regex, file_glob, deadline))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
