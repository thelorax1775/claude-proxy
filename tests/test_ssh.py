"""The SSH host registry and the argv it builds.

No network and no ssh binary needed: what matters here is that the registry
refuses input that would change what ssh connects to, and that the arguments
come out as separate argv elements rather than a string a shell could reopen.
"""

import os

import httpx
import pytest

import agent_tools
import main
import ssh_tools


@pytest.fixture
def hosts(app, tmp_path):
    """An empty registry backed by a temp file, restored afterwards."""
    original_path, original_hosts = ssh_tools.SSH_HOSTS_PATH, list(ssh_tools.HOSTS)
    ssh_tools.SSH_HOSTS_PATH = str(tmp_path / "ssh_hosts.json")
    ssh_tools.HOSTS.clear()
    yield ssh_tools
    ssh_tools.SSH_HOSTS_PATH = original_path
    ssh_tools.HOSTS[:] = original_hosts


@pytest.fixture
async def client(hosts):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Validation -- the registry is the allowlist, so this is where it earns it
# ---------------------------------------------------------------------------

def test_a_normal_host_validates(hosts):
    assert hosts.validate_host("nas", "192.168.1.10", "root", 22, "") == ""
    assert hosts.validate_host("box", "box.example.com") == ""


@pytest.mark.parametrize("name", [
    "",
    "-oProxyCommand=touch /tmp/pwned",   # would be read by ssh as an option
    "-bad",                              # a bare leading dash is the same problem
    "-oProxyCommand",                    # and needs no punctuation to be one
    "has space",
    "semi;colon",
    "x" * 65,
])
def test_bad_names_are_refused(hosts, name):
    assert hosts.validate_host(name, "example.com").startswith("[error]")


@pytest.mark.parametrize("host", [
    "",
    "-oProxyCommand=id",
    "-oProxyCommand",
    "example.com; rm -rf /",
    "user@example.com",       # the user belongs in its own field
    "example.com:22",         # so does the port
    "has space",
])
def test_bad_hostnames_are_refused(hosts, host):
    assert hosts.validate_host("ok", host).startswith("[error]")


@pytest.mark.parametrize("user", ["-oProxyCommand=id", "-root", "has space", "a;b"])
def test_bad_usernames_are_refused(hosts, user):
    assert hosts.validate_host("ok", "example.com", user).startswith("[error]")


@pytest.mark.parametrize("port", [0, 65536, -1, "nope"])
def test_bad_ports_are_refused(hosts, port):
    assert hosts.validate_host("ok", "example.com", "", port).startswith("[error]")


def test_key_path_must_be_absolute_and_not_an_option(hosts):
    assert hosts.validate_host("ok", "e.com", "", 22, "relative/key").startswith("[error]")
    assert hosts.validate_host("ok", "e.com", "", 22, "-oProxyCommand=id").startswith("[error]")
    assert hosts.validate_host("ok", "e.com", "", 22, "/home/me/.ssh/id_ed25519") == ""


def test_remote_paths_reject_newlines(hosts):
    assert hosts.check_remote_path("/tmp/ok") == ""
    assert hosts.check_remote_path("/tmp/a\nrm -rf /").startswith("[error]")
    assert hosts.check_remote_path("").startswith("[error]")


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------

def test_register_and_find(hosts):
    entry = hosts.register_host("nas", "192.168.1.10", "root", 2222, label="Storage")
    assert entry["name"] == "nas"
    assert entry["port"] == 2222
    assert hosts.find_host("nas") == entry
    assert hosts.find_host("nope") is None


def test_register_refuses_bad_input(hosts):
    assert hosts.register_host("-bad", "example.com").startswith("[error]")
    assert hosts.HOSTS == []


def test_registering_the_same_name_replaces_it(hosts):
    hosts.register_host("nas", "old.example.com")
    hosts.register_host("nas", "new.example.com")
    assert len(hosts.HOSTS) == 1
    assert hosts.find_host("nas")["host"] == "new.example.com"


def test_hosts_survive_a_reload(hosts):
    hosts.register_host("nas", "192.168.1.10", "root", 2222, label="Storage")
    reloaded = hosts.load_hosts()
    assert [h["name"] for h in reloaded] == ["nas"]
    assert reloaded[0]["port"] == 2222


def test_the_file_is_private(hosts):
    hosts.register_host("nas", "192.168.1.10")
    mode = os.stat(hosts.SSH_HOSTS_PATH).st_mode & 0o777
    assert mode == 0o600


def test_a_corrupt_file_starts_empty_rather_than_failing(hosts):
    with open(hosts.SSH_HOSTS_PATH, "w") as f:
        f.write("{not json")
    assert hosts.load_hosts() == []


def test_unusable_entries_in_the_file_are_skipped(hosts):
    import json
    with open(hosts.SSH_HOSTS_PATH, "w") as f:
        json.dump([{"name": "-bad", "host": "e.com"}, {"name": "good", "host": "e.com"}], f)
    assert [h["name"] for h in hosts.load_hosts()] == ["good"]


def test_public_host_never_leaks_a_key(hosts):
    entry = hosts.register_host("nas", "e.com", key_path="/nonexistent/key")
    public = hosts.public_host(entry)
    assert public["key_exists"] is False
    assert set(public) == {"name", "host", "user", "port", "label", "key_path", "key_exists"}


# ---------------------------------------------------------------------------
# argv construction
# ---------------------------------------------------------------------------

def test_connection_options_forbid_every_prompt(hosts):
    opts = hosts.base_opts({"key_path": ""})
    joined = " ".join(opts)
    # Any one of these missing turns a first-contact host key or a password
    # request into a tool call that hangs until the timeout.
    assert "BatchMode=yes" in joined
    assert "PasswordAuthentication=no" in joined
    assert "NumberOfPasswordPrompts=0" in joined
    assert "StrictHostKeyChecking=accept-new" in joined


def test_a_key_path_becomes_separate_argv_elements(hosts):
    opts = hosts.base_opts({"key_path": "/home/me/my key"})
    assert "-i" in opts
    # The path with its space is one element -- nothing re-splits it.
    assert opts[opts.index("-i") + 1] == "/home/me/my key"
    assert "IdentitiesOnly=yes" in " ".join(opts)


def test_target_and_describe(hosts):
    assert hosts.target({"user": "root", "host": "e.com"}) == "root@e.com"
    assert hosts.target({"user": "", "host": "e.com"}) == "e.com"
    assert hosts.describe({"user": "root", "host": "e.com", "port": 22}) == "root@e.com"
    assert hosts.describe({"user": "", "host": "e.com", "port": 2222}) == "e.com:2222"


def test_timeouts_are_clamped(hosts):
    assert hosts.clamp_timeout(None) == hosts.SSH_TIMEOUT
    assert hosts.clamp_timeout("nonsense") == hosts.SSH_TIMEOUT
    assert hosts.clamp_timeout(1) == 5
    assert hosts.clamp_timeout(10 ** 6) == hosts.SSH_MAX_TIMEOUT
    assert hosts.clamp_timeout(90) == 90


def test_bash_timeouts_are_clamped():
    assert agent_tools.clamp_tool_timeout(None) == agent_tools.AGENT_TOOL_TIMEOUT
    assert agent_tools.clamp_tool_timeout(10 ** 6) == agent_tools.BASH_MAX_TIMEOUT
    assert agent_tools.clamp_tool_timeout(120) == 120


# ---------------------------------------------------------------------------
# Group availability -- an ordinary toggle, gated on nothing
# ---------------------------------------------------------------------------

def test_the_group_is_available_with_no_hosts(hosts, monkeypatch):
    """An empty registry makes the tools useless, not unavailable. Disabling
    the checkbox until a host existed meant the Agent tab greyed out the
    control you were there to use."""
    monkeypatch.setattr(hosts, "ssh_available", lambda: True)
    assert agent_tools.available_groups()["ssh"] is True
    assert agent_tools.group_unavailable_reason("ssh") == ""


def test_the_group_is_available_without_an_ssh_binary(hosts, monkeypatch):
    """A missing ssh client is reported by the tool that tries to use it, the
    same way a missing git is -- it doesn't take the group away."""
    hosts.register_host("nas", "e.com")
    monkeypatch.setattr(hosts, "ssh_available", lambda: False)
    assert agent_tools.available_groups()["ssh"] is True
    assert agent_tools.group_unavailable_reason("ssh") == ""


def test_the_group_stays_available_with_a_host(hosts, monkeypatch):
    hosts.register_host("nas", "e.com")
    monkeypatch.setattr(hosts, "ssh_available", lambda: True)
    assert agent_tools.available_groups()["ssh"] is True
    assert agent_tools.group_unavailable_reason("ssh") == ""


def test_the_group_can_be_selected_with_an_empty_registry(hosts):
    """build_agent_context drops unusable groups, so 'available' has to hold
    for the tick to survive the round trip to the server."""
    ctx = agent_tools.build_agent_context(None, True, ["ssh"])
    assert ctx.allows("ssh")


def test_the_prompt_says_what_to_do_when_no_host_is_registered(hosts):
    ctx = agent_tools.build_agent_context(None, True, ["ssh"])
    prompt = agent_tools.agent_system_prompt(ctx)
    assert "none yet" in prompt and "Agent tab" in prompt


# ---------------------------------------------------------------------------
# known_hosts -- ssh will not record a key it cannot write
# ---------------------------------------------------------------------------

def test_known_hosts_defaults_under_the_users_ssh_dir(hosts):
    assert hosts.KNOWN_HOSTS.endswith(os.path.join(".ssh", "known_hosts"))


def test_the_options_always_name_the_known_hosts_file(hosts):
    joined = " ".join(hosts.base_opts({"key_path": ""}))
    assert f"UserKnownHostsFile={hosts.KNOWN_HOSTS}" in joined


def test_ensure_ssh_dir_creates_it_private(hosts, monkeypatch, tmp_path):
    target = tmp_path / "nonexistent" / "known_hosts"
    monkeypatch.setattr(hosts, "KNOWN_HOSTS", str(target))
    hosts.ensure_ssh_dir()
    assert target.parent.is_dir()
    assert target.parent.stat().st_mode & 0o777 == 0o700


def test_ensure_ssh_dir_survives_a_read_only_home(hosts, monkeypatch, tmp_path):
    """A home it cannot write is the connection's problem to report, not a
    reason to raise out of a tool call."""
    def boom(*a, **k):
        raise PermissionError("read-only file system")
    monkeypatch.setattr(hosts, "KNOWN_HOSTS", str(tmp_path / "nope" / "known_hosts"))
    monkeypatch.setattr(hosts.os, "makedirs", boom)
    hosts.ensure_ssh_dir()  # must not raise


def test_ssh_is_not_a_default_group():
    """It reaches a machine this proxy's sandbox has no authority over, so it
    is opt-in like the other two irreversible groups."""
    assert "ssh" not in agent_tools.DEFAULT_GROUPS
    assert "ssh" in agent_tools.GROUPS


async def test_an_unticked_ssh_group_is_refused_server_side(hosts, monkeypatch):
    hosts.register_host("nas", "e.com")
    monkeypatch.setattr(hosts, "ssh_available", lambda: True)
    ctx = agent_tools.build_agent_context(None, True, ["files"])
    result = await agent_tools.execute_tool_call(ctx, "ssh_run", {"host": "nas", "command": "id"})
    assert "switched off" in result


async def test_an_unknown_host_is_refused_with_the_list(hosts, monkeypatch):
    hosts.register_host("nas", "e.com")
    monkeypatch.setattr(hosts, "ssh_available", lambda: True)
    ctx = agent_tools.build_agent_context(None, True, ["ssh"])
    result = await agent_tools.execute_tool_call(ctx, "ssh_run",
                                                 {"host": "elsewhere", "command": "id"})
    assert result.startswith("[error]")
    assert "nas" in result          # tells the model what it may actually use


async def test_upload_respects_the_sandbox(hosts, monkeypatch, tmp_path):
    hosts.register_host("nas", "e.com")
    monkeypatch.setattr(hosts, "ssh_available", lambda: True)
    ctx = agent_tools.build_agent_context(None, True, ["ssh"])
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")

    result = await agent_tools.execute_tool_call(ctx, "ssh_upload", {
        "host": "nas", "local_path": str(outside), "remote_path": "/tmp/x"})
    assert "outside this session's working directories" in result


async def test_download_refuses_to_overwrite(hosts, monkeypatch):
    hosts.register_host("nas", "e.com")
    monkeypatch.setattr(hosts, "ssh_available", lambda: True)
    ctx = agent_tools.build_agent_context(None, True, ["ssh"])
    existing = os.path.join(ctx.cwd, "already-here.txt")
    with open(existing, "w") as f:
        f.write("mine")
    try:
        result = await agent_tools.execute_tool_call(ctx, "ssh_download", {
            "host": "nas", "remote_path": "/tmp/x", "local_path": "already-here.txt"})
        assert "already exists" in result
    finally:
        os.remove(existing)


async def test_ssh_list_hosts_names_them(hosts, monkeypatch):
    hosts.register_host("nas", "192.168.1.10", "root", 2222, label="Storage")
    monkeypatch.setattr(hosts, "ssh_available", lambda: True)
    ctx = agent_tools.build_agent_context(None, True, ["ssh"])
    result = await agent_tools.execute_tool_call(ctx, "ssh_list_hosts", {})
    assert "nas" in result and "root@192.168.1.10:2222" in result


def test_the_system_prompt_names_the_hosts(hosts, monkeypatch):
    hosts.register_host("nas", "e.com")
    monkeypatch.setattr(hosts, "ssh_available", lambda: True)
    ctx = agent_tools.build_agent_context(None, True, ["ssh"])
    prompt = agent_tools.agent_system_prompt(ctx)
    assert "nas" in prompt
    assert "DIFFERENT machine" in prompt


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

async def test_crud_round_trip(client):
    resp = await client.post("/api/ssh-hosts", json={
        "name": "nas", "host": "192.168.1.10", "user": "root", "port": 2222})
    assert resp.status_code == 200
    assert resp.json()["name"] == "nas"

    listed = (await client.get("/api/ssh-hosts")).json()
    assert [h["name"] for h in listed["hosts"]] == ["nas"]
    assert "ssh_available" in listed

    assert (await client.delete("/api/ssh-hosts?name=nas")).status_code == 200
    assert (await client.get("/api/ssh-hosts")).json()["hosts"] == []


async def test_the_api_refuses_a_hostile_entry(client):
    resp = await client.post("/api/ssh-hosts", json={
        "name": "-oProxyCommand=touch /tmp/pwned", "host": "e.com"})
    assert resp.status_code == 400
    assert (await client.get("/api/ssh-hosts")).json()["hosts"] == []


async def test_deleting_an_unknown_host_is_404(client):
    assert (await client.delete("/api/ssh-hosts?name=nope")).status_code == 404


async def test_testing_an_unknown_host_is_404(client):
    assert (await client.post("/api/ssh-hosts/nope/test")).status_code == 404


async def test_settings_reports_the_ssh_group_as_available(client):
    settings = (await client.get("/api/settings")).json()
    group = next(g for g in settings["groups"] if g["name"] == "ssh")
    assert group["available"] is True
    assert not group["reason"]
    assert set(group["tools"]) == {"ssh_list_hosts", "ssh_run", "ssh_upload", "ssh_download"}
    assert settings["ssh_timeout"] and settings["ssh_max_timeout"]
