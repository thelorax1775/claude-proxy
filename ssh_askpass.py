#!/usr/bin/env python3
"""Answer ssh's password prompt from the environment, and nothing else.

OpenSSH will ask a helper program for a passphrase when SSH_ASKPASS names one
and SSH_ASKPASS_REQUIRE=force says to use it even without a terminal. That is
the whole mechanism behind password auth here: no pty to drive, no sshpass to
install, no second SSH implementation to keep in step with the first -- the
proxy still runs the system ssh and scp binaries exactly as it does for keys.

ssh runs this with the prompt it wants answered as argv[1]. Two rules:

  * Only a password prompt is answered. ssh uses the same helper to ask "are
    you sure you want to continue connecting" on an unrecognised host key, and
    a helper that blindly prints a secret would hand it to whatever answered
    the connection. StrictHostKeyChecking should mean that prompt never
    appears; this is the belt to that braces.

  * The password arrives in the environment, which is readable through
    /proc only by this user and root -- the same exposure sshpass -e has, and
    strictly better than an argv element, which every process on the machine
    can read.

Exiting non-zero makes ssh treat the prompt as unanswered and fail, which is
the right outcome for anything unexpected: a hung tool call with no output is
the failure mode this whole file exists to avoid.
"""

import os
import sys

ENV_VAR = "CLAUDE_PROXY_SSH_PASSWORD"

# ssh's own wording varies by version and by whether the server asked for a
# password or a keyboard-interactive challenge. Matching on "password" and
# "passcode" covers both without matching the host-key question, which is
# phrased as "Are you sure you want to continue connecting (yes/no...)".
WANTED = ("password", "passcode", "passphrase")


def main() -> int:
    prompt = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if not any(word in prompt for word in WANTED):
        # A yes/no confirmation, or something new. Refuse rather than guess.
        print(f"claude-proxy askpass: refusing to answer {prompt!r}", file=sys.stderr)
        return 1
    password = os.environ.get(ENV_VAR)
    if password is None:
        print("claude-proxy askpass: no password in the environment", file=sys.stderr)
        return 1
    sys.stdout.write(password + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
