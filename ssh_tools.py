"""SSH for agent mode: a registry of hosts, and the ssh/scp mechanics.

The registry is the point. Tools take a registered host *name*, never a
connection string, so the set of machines the model can reach is something a
person put there by hand -- the same shape as the working-directory registry,
where a session picks from a list rather than inventing entries.

Standard library plus the shared subprocess helper only, and no new package:
this drives the system `ssh`/`scp` binaries through create_subprocess_exec, so
a hostname or a path is an argv element and can never be reinterpreted by a
local shell.

A host authenticates with a key or with a password. Keys are better and stay
the default; a password is there because plenty of machines -- a router, an
appliance, a NAS, someone else's box -- will never have a key installed on
them, and "use a key" is not advice you can act on for those.

A stored password is answered by ssh_askpass.py, so the system ssh and scp
binaries still do all the work; see base_opts() for how prompting is opened up
for exactly one attempt on exactly the hosts that need it. The password lives
in ssh_hosts.json at 0600, next to the machines it belongs to. That is real
plaintext-at-rest, and the reason the field is optional and the docs say to
prefer a key.
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

# Where ssh keeps the host keys it has accepted. Defaults to the ordinary
# ~/.ssh/known_hosts; override only to put it somewhere else.
#
# It is spelled out rather than left to ssh because ssh will not record a key
# it cannot write, and a service account frequently has no home directory on
# disk at all -- the account is created for the unit, nobody ever logs in as
# it, so nothing ever makes the directory. Naming the path lets ensure_ssh_dir
# create it once, up front, instead of every first connection failing the
# host-key check with a permissions error that reads like a network problem.
KNOWN_HOSTS = (os.getenv("AGENT_SSH_KNOWN_HOSTS", "").strip()
               or os.path.join(os.path.expanduser("~"), ".ssh", "known_hosts"))

# The helper ssh calls to get a stored password. Alongside this file rather
# than found on PATH: it is part of the proxy, and resolving it by name would
# mean whatever is first on PATH gets handed a password.
ASKPASS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssh_askpass.py")

# Some servers ask for a password over keyboard-interactive rather than the
# password method, and a lot of appliances only offer one of the two. Naming
# both, and nothing else, also stops ssh burning its one allowed attempt on a
# public key that was never going to work.
PASSWORD_AUTHS = "password,keyboard-interactive"


def ensure_ssh_dir() -> None:
    """Make sure the directory holding known_hosts exists, 0700. Best effort:
    a read-only home is a reason for the connection to fail with ssh's own
    message, not a reason for the proxy to refuse to start."""
    parent = os.path.dirname(KNOWN_HOSTS)
    if not parent or os.path.isdir(parent):
        return
    try:
        os.makedirs(parent, mode=0o700, exist_ok=True)
    except OSError as e:
        log.warning("could not create %s (%s) -- ssh may fail the host-key check "
                    "until it exists and is writable", parent, e)


def ensure_askpass() -> str:
    """Make sure the askpass helper is executable, and say so if it isn't.

    ssh execs this file itself, so the mode matters. It is checked at use
    rather than trusted from the checkout because the deploy step here is an
    archive extraction, and a helper that is merely present but not executable
    fails as "permission denied" from the far end -- an error that reads like a
    wrong password and sends you looking in entirely the wrong place.
    """
    if not os.path.isfile(ASKPASS):
        return f"[error] the password helper is missing from the install ({ASKPASS})"
    if os.access(ASKPASS, os.X_OK):
        return ""
    try:
        os.chmod(ASKPASS, 0o755)
    except OSError as e:
        return f"[error] the password helper is not executable and could not be fixed: {e}"
    return ""


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

def validate_password(password: str) -> str:
    """Empty string if this password can be delivered to ssh, an [error] line
    if not.

    A newline would end the answer early and leave the rest to be read as the
    next thing ssh asks for; a NUL cannot survive the environment at all. Both
    would fail at connect time as a baffling auth error, so they are refused
    here where there is somebody to tell."""
    if not password:
        return ""
    if any(c in password for c in ("\n", "\r", "\0")):
        return "[error] a password cannot contain newlines or null bytes"
    if len(password) > 1024:
        return "[error] that password is implausibly long (over 1024 characters)"
    return ""


def validate_host(name: str, host: str, user: str = "", port=22, key_path: str = "",
                  password: str = "") -> str:
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
    problem = validate_password(password)
    if problem:
        return problem
    # ssh needs somebody to be the user when a password is involved: with a key
    # it can fall back to the local account name, but a password prompt for an
    # unstated user is a guess nobody can debug from the error.
    if password and not user:
        return "[error] a username is required when using a password"
    return ""


def _clean(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    host = str(item.get("host") or "").strip()
    user = str(item.get("user") or "").strip()
    key_path = str(item.get("key_path") or "").strip()
    # Not stripped: leading and trailing whitespace can be part of a password,
    # and silently trimming one is an auth failure nobody would think to look
    # for here.
    password = item.get("password") or ""
    if not isinstance(password, str):
        password = ""
    try:
        port = int(item.get("port") or 22)
    except (TypeError, ValueError):
        port = 22
    if validate_host(name, host, user, port, key_path, password):
        return None
    return {
        "name": name,
        "host": host,
        "user": user,
        "port": port,
        "key_path": os.path.expanduser(key_path) if key_path else "",
        "password": password,
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
                  key_path: str = "", label: str = "",
                  password: str | None = None) -> dict | str:
    """Add or replace a host. Returns the entry, or an [error] string.

    `password` is three-valued, because editing a host must not require
    retyping its password and clearing one must still be possible:
    None keeps whatever is saved, "" removes it, anything else replaces it.
    """
    name, host, user = (name or "").strip(), (host or "").strip(), (user or "").strip()
    key_path = (key_path or "").strip()
    existing = find_host(name)
    if password is None:
        password = existing.get("password", "") if existing else ""
    problem = validate_host(name, host, user, port, key_path, password)
    if problem:
        return problem

    entry = {
        "name": name,
        "host": host,
        "user": user,
        "port": int(port or 22),
        "key_path": os.path.expanduser(key_path) if key_path else "",
        "password": password,
        "label": (label or "").strip() or name,
    }
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
    itself, and never the password -- only that one is set. This is the single
    place a host becomes JSON, so a secret that isn't named here cannot reach
    a response, a chat transcript, or a tool result."""
    return {
        "name": h["name"],
        "host": h["host"],
        "user": h["user"],
        "port": h["port"],
        "label": h["label"],
        "key_path": h["key_path"],
        "key_exists": bool(h["key_path"]) and os.path.isfile(h["key_path"]),
        "has_password": bool(h.get("password")),
        "auth": "password" if h.get("password") else "key",
    }


# ---------------------------------------------------------------------------
# Running things
# ---------------------------------------------------------------------------

def base_opts(h: dict) -> list[str]:
    """Options common to ssh and scp, built only from registry values.

    The constant across both auth modes is that ssh must never stop and wait
    for a human. There is nobody at a terminal, so a prompt is not a question,
    it is a tool call that hangs until the timeout with no clue as to why.

    Without a password that means refusing to prompt at all. With one it means
    the opposite -- prompting has to be possible for the helper to answer --
    so the guarantee is re-established a different way: exactly one attempt,
    and only for the auth methods a password can actually satisfy. A wrong
    password comes back as a permission error in a second rather than three
    retries and a wait.

    StrictHostKeyChecking is untouched by any of this. A password does not
    make an unrecognised host key acceptable; if anything it raises the stakes,
    since a key is never sent to the far end and a password is.
    """
    opts = [
        "-o", f"StrictHostKeyChecking={STRICT_HOST_KEY}",
        "-o", "ConnectTimeout=10",
    ]
    if h.get("password"):
        opts += [
            "-o", "BatchMode=no",
            "-o", "PasswordAuthentication=yes",
            "-o", "KbdInteractiveAuthentication=yes",
            "-o", "NumberOfPasswordPrompts=1",
            "-o", f"PreferredAuthentications={PASSWORD_AUTHS}",
        ]
    else:
        opts += [
            "-o", "BatchMode=yes",
            "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no",
            "-o", "NumberOfPasswordPrompts=0",
        ]
    if KNOWN_HOSTS:
        opts += ["-o", f"UserKnownHostsFile={KNOWN_HOSTS}"]
    if h.get("key_path"):
        # IdentitiesOnly stops the agent offering a dozen other keys first,
        # which some servers count as failed attempts and disconnect over.
        opts += ["-i", h["key_path"], "-o", "IdentitiesOnly=yes"]
    return opts


def child_env(h: dict) -> dict | None:
    """The environment for this host's ssh, or None to inherit unchanged.

    Only a password host gets a modified environment, and the password is put
    there rather than in argv because argv is world-readable through /proc and
    the environment is not. SSH_ASKPASS_REQUIRE=force is what makes ssh consult
    the helper with no terminal and no DISPLAY -- without it the variable is
    ignored and the connection fails on a prompt nobody answers.
    """
    if not h.get("password"):
        return None
    env = dict(os.environ)
    env["SSH_ASKPASS"] = ASKPASS
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["CLAUDE_PROXY_SSH_PASSWORD"] = h["password"]
    return env


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


async def _run(program: str, args: list, timeout: int,
               env: dict | None = None) -> tuple[int | None, str]:
    """Shared exec, using git_tools' shielded kill-and-reap so a timeout or a
    Stop press never leaves an ssh session running behind a UI that says the
    run has finished."""
    import asyncio

    proc = await asyncio.create_subprocess_exec(
        program, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        env=env,
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
    ensure_ssh_dir()
    if h.get("password"):
        problem = ensure_askpass()
        if problem:
            return problem
    args = [*base_opts(h), "-p", str(h["port"]), "--", target(h), command]
    try:
        rc, out = await _run("ssh", args, timeout, env=child_env(h))
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
    ensure_ssh_dir()
    if h.get("password"):
        problem = ensure_askpass()
        if problem:
            return problem
    remote_spec = f"{target(h)}:{shlex.quote(remote)}"
    src, dst = (local, remote_spec) if upload else (remote_spec, local)
    args = [*base_opts(h), "-P", str(h["port"]), "--", src, dst]
    try:
        rc, out = await _run("scp", args, timeout, env=child_env(h))
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
