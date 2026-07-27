"""SSH for agent mode: a registry of hosts, and the ssh/scp mechanics.

The registry is the point. Tools take a registered host *name*, never a
connection string, so the set of machines the model can reach is something a
person put there by hand -- the same shape as the working-directory registry,
where a session picks from a list rather than inventing entries.

Standard library plus the shared subprocess helper only, and no new package:
this drives the system `ssh`/`scp` binaries through create_subprocess_exec, so
a hostname or a path is an argv element and can never be reinterpreted by a
local shell.

Auth is key-based. No password is stored, accepted, or promptable -- see
base_opts() for why that is enforced rather than merely expected.
"""

import json
import logging
import os
import re
import shlex
import shutil

import git_tools

log = logging.getLogger("claude-proxy")

SSH_HOSTS_PATH = os.getenv("SSH_HOSTS_CONFIG", "ssh_hosts.json")

# AGENT_TOOL_TIMEOUT (30s) is tuned for a local command. A remote build, a
# package update or a restart is a normal thing to ask for and routinely takes
# longer, so the ssh tools get their own budget and an optional per-call
# override bounded by the ceiling.
SSH_TIMEOUT = int(os.getenv("AGENT_SSH_TIMEOUT", "120"))
SSH_MAX_TIMEOUT = int(os.getenv("AGENT_SSH_MAX_TIMEOUT", "600"))

# accept-new records a host key on first contact but still refuses a *changed*
# one. "no" would silently accept a substituted key, which is to say accept a
# machine-in-the-middle; "yes" makes first contact fail until someone hand-edits
# known_hosts. Overridable for anyone who pre-populates known_hosts and wants
# the stricter setting.
STRICT_HOST_KEY = os.getenv("AGENT_SSH_STRICT_HOST_KEY", "accept-new")

# Where ssh keeps the host keys it has accepted. Empty means ssh's own default
# (~/.ssh/known_hosts), which is right almost everywhere -- but under a
# hardened systemd unit ProtectHome=true hides the service user's home, and
# ssh then cannot record a host key at all. Pointing this at a writable path
# inside the unit's own tree is the fix.
KNOWN_HOSTS = os.getenv("AGENT_SSH_KNOWN_HOSTS", "").strip()

# Deliberately narrow. These values become argv elements for ssh, where a
# leading "-" is an option rather than a hostname, and where whitespace or an
# embedded @ or : changes what is being connected to. The first character is
# spelled out separately so a leading dash can't slip through the class --
# "-oProxyCommand" is otherwise a perfectly ordinary-looking hostname.
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$")
USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$")
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")

OUTPUT_LIMIT = 8000


def ssh_available() -> bool:
    """Whether there is an ssh client to drive at all. Reported through
    /api/settings so the group can say why it's unavailable instead of
    offering tools that can only fail."""
    return bool(shutil.which("ssh"))


def clamp_timeout(requested, default: int = SSH_TIMEOUT) -> int:
    try:
        value = int(requested)
    except (TypeError, ValueError):
        return default
    return max(5, min(value, SSH_MAX_TIMEOUT))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

def validate_host(name: str, host: str, user: str = "", port=22, key_path: str = "") -> str:
    """Empty string if this is a usable entry, an [error] line if not.
    Everything is checked here, on the way in, rather than at use: a name that
    ssh would read as an option must never reach the registry in the first
    place."""
    if not NAME_RE.match(name or ""):
        return ("[error] the name must be 1-64 characters of letters, digits, dot, dash "
                "or underscore, and must not start with a dash")
    if not HOST_RE.match(host or ""):
        return f"[error] '{host}' is not a usable hostname or address"
    if user and not USER_RE.match(user):
        return f"[error] '{user}' is not a usable username"
    try:
        port = int(port)
    except (TypeError, ValueError):
        return "[error] port must be a number"
    if not 1 <= port <= 65535:
        return "[error] port must be between 1 and 65535"
    if key_path:
        expanded = os.path.expanduser(key_path)
        if expanded.startswith("-"):
            return "[error] the key path must not start with '-'"
        if not os.path.isabs(expanded):
            return "[error] the key path must be absolute"
    return ""


def _clean(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    host = str(item.get("host") or "").strip()
    user = str(item.get("user") or "").strip()
    key_path = str(item.get("key_path") or "").strip()
    try:
        port = int(item.get("port") or 22)
    except (TypeError, ValueError):
        port = 22
    if validate_host(name, host, user, port, key_path):
        return None
    return {
        "name": name,
        "host": host,
        "user": user,
        "port": port,
        "key_path": os.path.expanduser(key_path) if key_path else "",
        "label": str(item.get("label") or "").strip() or name,
    }


def load_hosts() -> list[dict]:
    """Tolerant, like load_workdirs: an unreadable or half-written file means
    starting empty with a warning, not refusing to boot."""
    if not os.path.exists(SSH_HOSTS_PATH):
        return []
    try:
        with open(SSH_HOSTS_PATH) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        log.warning("could not read %s (%s) -- starting with no SSH hosts", SSH_HOSTS_PATH, e)
        return []

    hosts, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        entry = _clean(item)
        if entry is None:
            log.warning("skipping an unusable entry in %s: %r", SSH_HOSTS_PATH, item)
            continue
        if entry["name"] in seen:
            continue
        seen.add(entry["name"])
        hosts.append(entry)
    return hosts


HOSTS = load_hosts()


def save_hosts() -> None:
    tmp = f"{SSH_HOSTS_PATH}.tmp"
    # 0600 from the moment it exists. Hostnames and usernames aren't secrets
    # the way a token is, but this file is a list of machines that can be
    # reached from here, which is not something to leave world-readable.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass  # best effort -- Windows and some mounts don't do POSIX modes
    with os.fdopen(fd, "w") as f:
        json.dump(HOSTS, f, indent=2)
    os.replace(tmp, SSH_HOSTS_PATH)


def find_host(name: str) -> dict | None:
    return next((h for h in HOSTS if h["name"] == name), None)


def register_host(name: str, host: str, user: str = "", port=22,
                  key_path: str = "", label: str = "") -> dict | str:
    """Add or replace a host. Returns the entry, or an [error] string."""
    name, host, user = (name or "").strip(), (host or "").strip(), (user or "").strip()
    key_path = (key_path or "").strip()
    problem = validate_host(name, host, user, port, key_path)
    if problem:
        return problem

    entry = {
        "name": name,
        "host": host,
        "user": user,
        "port": int(port or 22),
        "key_path": os.path.expanduser(key_path) if key_path else "",
        "label": (label or "").strip() or name,
    }
    existing = find_host(name)
    if existing is not None:
        HOSTS[HOSTS.index(existing)] = entry
    else:
        HOSTS.append(entry)
    save_hosts()
    return entry


def remove_host(name: str) -> bool:
    entry = find_host(name)
    if entry is None:
        return False
    HOSTS.remove(entry)
    save_hosts()
    return True


def public_host(h: dict) -> dict:
    """What the browser and the model are allowed to see. Never the key
    itself -- only whether the configured path exists."""
    return {
        "name": h["name"],
        "host": h["host"],
        "user": h["user"],
        "port": h["port"],
        "label": h["label"],
        "key_path": h["key_path"],
        "key_exists": bool(h["key_path"]) and os.path.isfile(h["key_path"]),
    }


# ---------------------------------------------------------------------------
# Running things
# ---------------------------------------------------------------------------

def base_opts(h: dict) -> list[str]:
    """Options common to ssh and scp, built only from registry values.

    BatchMode/PasswordAuthentication/NumberOfPasswordPrompts together mean ssh
    can never stop at a prompt. Without them an unknown host key or a password
    request turns into a tool call that just hangs until the timeout, with no
    indication of why -- there is nobody at a terminal to answer."""
    opts = [
        "-o", "BatchMode=yes",
        "-o", f"StrictHostKeyChecking={STRICT_HOST_KEY}",
        "-o", "PasswordAuthentication=no",
        "-o", "KbdInteractiveAuthentication=no",
        "-o", "NumberOfPasswordPrompts=0",
        "-o", "ConnectTimeout=10",
    ]
    if KNOWN_HOSTS:
        opts += ["-o", f"UserKnownHostsFile={KNOWN_HOSTS}"]
    if h.get("key_path"):
        # IdentitiesOnly stops the agent offering a dozen other keys first,
        # which some servers count as failed attempts and disconnect over.
        opts += ["-i", h["key_path"], "-o", "IdentitiesOnly=yes"]
    return opts


def target(h: dict) -> str:
    return f"{h['user']}@{h['host']}" if h.get("user") else h["host"]


def describe(h: dict) -> str:
    port = "" if h["port"] == 22 else f":{h['port']}"
    return f"{target(h)}{port}"


def _truncate(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    return text[:OUTPUT_LIMIT] + f"\n...[truncated, {len(text)} bytes total]"


def check_remote_path(path: str) -> str:
    """Remote paths are interpolated into a remote shell command by ssh
    itself, so they get quoted at the call site -- but a newline or a NUL
    would still break the command apart before quoting can help."""
    if not path:
        return "[error] a remote path is required"
    if "\n" in path or "\r" in path or "\0" in path:
        return "[error] a remote path cannot contain newlines"
    return ""


async def _run(program: str, args: list, timeout: int) -> tuple[int | None, str]:
    """Shared exec, using git_tools' shielded kill-and-reap so a timeout or a
    Stop press never leaves an ssh session running behind a UI that says the
    run has finished."""
    import asyncio

    proc = await asyncio.create_subprocess_exec(
        program, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await git_tools.stop_process(proc)
        return None, ""
    except BaseException:
        await git_tools.stop_process(proc)
        raise
    return proc.returncode, out.decode("utf-8", "replace")


async def run_command(h: dict, command: str, timeout: int) -> str:
    """Run one command on a host and return its combined output."""
    if not command or not command.strip():
        return "[error] a command is required"
    args = [*base_opts(h), "-p", str(h["port"]), "--", target(h), command]
    try:
        rc, out = await _run("ssh", args, timeout)
    except FileNotFoundError:
        return "[error] ssh is not installed on the proxy's machine"
    except OSError as e:
        return f"[error] could not run ssh: {e}"

    if rc is None:
        return (f"[error] the command timed out after {timeout}s on {h['name']}. "
                "Pass a larger 'timeout' if it genuinely needs longer.")
    text = _truncate(out)
    if rc != 0:
        return f"[exit {rc}] on {h['name']} ({describe(h)})\n{text or '(no output)'}"
    return text or "(no output)"


async def copy_file(h: dict, local: str, remote: str, *, upload: bool, timeout: int) -> str:
    """scp one file to or from a host. Note scp spells the port -P, not -p."""
    remote_spec = f"{target(h)}:{shlex.quote(remote)}"
    src, dst = (local, remote_spec) if upload else (remote_spec, local)
    args = [*base_opts(h), "-P", str(h["port"]), "--", src, dst]
    try:
        rc, out = await _run("scp", args, timeout)
    except FileNotFoundError:
        return "[error] scp is not installed on the proxy's machine"
    except OSError as e:
        return f"[error] could not run scp: {e}"

    if rc is None:
        return f"[error] the transfer timed out after {timeout}s"
    if rc != 0:
        return f"[error] scp failed [exit {rc}]\n{_truncate(out) or '(no output)'}"
    direction = "to" if upload else "from"
    return f"copied {local} {direction} {h['name']}:{remote}"
