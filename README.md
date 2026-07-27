# claude-proxy

Runs Claude Code against a model of your choosing instead of Anthropic, by
translating between Anthropic's Messages API (what Claude Code speaks) and
any OpenAI-compatible chat completions API -- with automatic failover
across a list of backends.

**What this actually is:** Claude Code stays Claude Code (the CLI/tool-use
harness). The *model* answering prompts is whichever backend in
`backends.json` actually responds. You are not getting Anthropic's Claude
models for free; you're using Claude Code as a CLI frontend for other
models.

There's also a web UI bundled in, which is a full chat client in its own
right with an agent mode that can work on real files, git repositories and
GitHub. That part is separate from Claude Code and doesn't require it.

---

## Quickstart

Four steps. Everything after this section is detail on one of them.

```bash
# 1. Install
git clone https://github.com/thelorax1775/claude-proxy
cd claude-proxy
pip install -r requirements.txt
cp backends.example.json backends.json

# 2. Run it
./start.sh                      # or ./start.ps1 on Windows
```

**3. Add a backend.** Open <http://127.0.0.1:8082/> → **Backends** tab →
pick a provider preset → paste its API key → **Detect models** → save. No
JSON editing needed. See [Configure backends](#configure-backends) for where
to get free keys.

**4. Point Claude Code at it.** Open the **CLI** tab in that same UI. It
generates the exact config block for you, with your address already filled
in. Save it as `~/.claude/settings.json`, then run `claude` from anywhere.

That's it. `curl http://127.0.0.1:8082/health` should return
`{"status":"ok",...}`, and the **Backends** tab's "Recent attempts" shows
which backend served each request.

### Requirements

- [Claude Code](https://github.com/anthropics/claude-code)
  (`npm install -g @anthropic-ai/claude-code`) -- only if you want the CLI;
  the web UI works without it
- Python 3.10+
- At least one OpenAI-compatible API to talk to: a free-tier account
  (Groq, Google AI Studio, Cerebras, OpenRouter, NVIDIA NIM), and/or your
  own Ollama instance

---

## Point Claude Code at it

Claude Code decides where to send requests from two environment variables.
You can set them once in a config file, or per terminal.

### Recommended: set it once

Save this as `~/.claude/settings.json` and every `claude` session on the
machine uses the proxy -- every project, every terminal, no exports:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8082",
    "ANTHROPIC_API_KEY": "local-claude-proxy"
  }
}
```

- On Windows that path is `%USERPROFILE%\.claude\settings.json`.
- For one project only, use `.claude/settings.json` inside that project.
- **If the file already exists, merge the `env` block in** rather than
  replacing the file.
- Claude Code reads it at launch.

The **CLI tab in the web UI generates this for you** with the right address
already substituted, plus a copy button -- which is worth using, because the
correct `ANTHROPIC_BASE_URL` is whatever address *the machine running
`claude`* can reach the proxy on, and that isn't always `127.0.0.1`.

`ANTHROPIC_API_KEY` is an unchecked placeholder. The proxy has no auth and
never looks at it; Claude Code just refuses to start without something set.
Your real provider keys live in `backends.json` on the proxy's machine.

### Alternative: per terminal

Applies only to the shell you run it in, and has to be repeated in every new
one. Fine for a quick test.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=local-claude-proxy
claude
```

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8082"
$env:ANTHROPIC_API_KEY = "local-claude-proxy"
claude
```

### If `claude` runs on a different machine

Set `PROXY_HOST` in `.env` to an interface the proxy is actually reachable
on (a Tailscale or LAN IP), restart, and use that address as
`ANTHROPIC_BASE_URL`. There is no auth in front of this proxy, so never bind
it to a public interface -- see [No auth, by design](#no-auth-by-design).

### Which model actually answers

Claude Code sends whatever model string is active (its default, or whatever
you set `ANTHROPIC_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` to). Per backend,
the proxy buckets on the string:

- contains `haiku` → that backend's `small_model`
- everything else (sonnet, opus, default) → that backend's `big_model`

Change the mapping in `backends.json`, not in code.

**Every model you list must support function/tool calling.** Claude Code's
file editing and bash execution run through Anthropic tool-use blocks,
translated here into OpenAI `tools`/`tool_calls`. A model without tool
support won't error loudly -- Claude Code will just seem to "forget" how to
edit files whenever failover lands on it. Check the provider's docs before
adding a model.

---

## Configure backends

`backends.json` is an ordered list. On every request the proxy tries backend
#1; if it gets rate-limited (429), errors (5xx), or its key is dead
(401/403), it logs that and tries backend #2, and so on -- ideally down to a
local Ollama instance last, which has no quota to exhaust. First backend that
answers wins.

Watch it happen live in the **Backends** tab, or:

```bash
curl http://127.0.0.1:8082/status
```

The **Backends** tab is the easiest way to edit this: provider presets, a
**Detect models** button that pulls the real model list from the provider
instead of you typing model IDs by hand, and drag-free reordering. It writes
`backends.json` for you, so keys never go through a chat transcript.

### Where to get free keys

| Provider | Where | Notes |
| --- | --- | --- |
| **Groq** | console.groq.com → API Keys | Generous free tier, fast, good tool-calling |
| **Google AI Studio** | aistudio.google.com → Get API key | Free tier, OpenAI-compatible endpoint |
| **Cerebras** | cloud.cerebras.ai → API Keys | Free tier |
| **OpenRouter** | openrouter.ai/keys | `:free` models cost nothing (rate-limited) |
| **NVIDIA NIM** | build.nvidia.com → API Keys | Trial credits, not unlimited |
| **Local Ollama** | — | No key. Point `base_url` at e.g. `http://localhost:11434/v1` |

> **Read this once:** sign up for one account each at *different* services,
> not multiple accounts on the *same* service. Free tiers are one-per-person
> by design, and stacking accounts to multiply one provider's quota is
> against most of their terms -- it risks getting all those keys banned
> together, breaking the whole chain at once. Five different providers gets
> you the same resilience with no such risk.

You don't need every provider filled in -- fewer entries just means a
shorter failover chain. Keep Ollama last so there's always a backend that
can't run out.

Make sure any Ollama model tag you list is actually pulled
(`ollama pull llama3.1:8b`) and supports tool calling. Model IDs in
`backends.example.json` were current as of this project's build date;
providers rename and revise models, so if a backend keeps failing, check its
console for the current slug before assuming something else is broken.

### If Ollama is behind Open WebUI

If you've deliberately bound Ollama to `127.0.0.1` and Open WebUI is the only
exposed gateway, point `base_url` at Open WebUI instead:
`http://<host>:8080/api`, with an `Authorization: Bearer <key>` generated in
its Settings → Account → API Keys. Its OpenAI-compatible route is
`POST /api/chat/completions`, not `/v1/chat/completions` -- that path serves
Open WebUI's own frontend on most versions. This proxy's
`{base_url}/chat/completions` construction handles both cases as long as
`base_url` includes whichever prefix your setup needs.

---

## Web UI

Open `http://127.0.0.1:8082/` (or wherever you bound it). A side panel sits
next to a chat pane, with the mode switches directly above the composer.

- **Chats** — every conversation is saved server-side under `sessions/`:
  reopen, rename, delete. Reopening replays the whole transcript including
  tool calls and their output, not just the text, and carries on with the
  same history. Survives a browser reload and a proxy restart.
- **Backends** — see [Configure backends](#configure-backends).
- **Agent** — tool-round budget, tool groups, working directories, web-search
  key. See [Agent tools](#agent-tools).
- **GitHub** — connect a token, browse repositories, clone one onto the
  proxy's machine with a click, read PRs and issues. The **→** button on any
  PR or issue drops a ready-made prompt into the composer.
- **CLI** — generates your `settings.json` for Claude Code, with the address
  filled in. See [Point Claude Code at it](#point-claude-code-at-it).
- **Chat** — talk to any configured backend directly, with streaming. This is
  a human talking to a provider; no Anthropic translation happens here, which
  is what makes it a useful way to test a backend in isolation. If that
  backend gets rate-limited or its key dies mid-conversation, the chat falls
  through the rest of your configured backends with the same history, and a
  note announces the switch. **Send** becomes **Stop** while streaming.

API keys run through this local server only -- they never leave the machine
the proxy runs on except to reach the provider you picked.

### No auth, by design

There is no login and no shared key: **anything that can reach the port can
spend your API keys, and can switch on agent mode and get a shell on the
machine running the proxy.** `PROXY_HOST` is the only thing limiting that.
Keep it on `127.0.0.1`, or on a private interface (e.g. a Tailscale IP) that
only you can reach. Never expose this port to the internet.

---

## Agent tools

Lets the model in the web UI's chat actually work on the machine hosting the
proxy: run commands, read and edit files, drive git, use GitHub, and fetch
from the web. This is **separate from Claude Code**, which has its own tools
and is unaffected by any of it. Everything is controlled from the UI -- no
env var to set first, no restart.

### The three switches

All sit above the composer and are remembered per browser:

- **Agent tools** — off means a plain chat with no tools at all. On means each
  message can spend a budget of tool rounds working before it answers (100 by
  default, see [Tool rounds](#tool-rounds)).
- **Sandbox** — on (the default) confines the file tools to the working
  directories ticked in the Agent tab: relative paths resolve against the
  primary one, absolute paths outside them are refused. Off lets the file
  tools reach anything the proxy's user can read or write.
- **LAN access** — off (the default) means `web_fetch` reaches the public
  internet only; private, loopback and link-local addresses are refused. On
  lets it reach this machine's own network. Locked unless the operator has
  allowed it — see [Reaching the local network](#reaching-the-local-network).

The chip next to the model dropdown always shows the live state, e.g.
`agent · sandboxed · myproject +1 · LAN`.

### Working directories

The Agent tab lists every directory agent mode may work in. `workspace/`
(from `AGENT_WORKSPACE`, created automatically) is always there and can't be
removed; add any other directory on the proxy's machine by typing its path.
Tick the ones the current session should see -- the first ticked directory is
the primary one, shown as `primary` on its card and used as `bash`'s working
directory.

Added directories are stored in `workdirs.json` (gitignored) and survive a
restart. Ticking is per browser, not shared.

> Under the systemd unit, `ProtectSystem=strict` means only the paths in
> `ReadWritePaths=` are writable. If you add a working directory outside
> `/opt/claude-proxy` and the agent can't write to it, add it there too and
> `daemon-reload` — and set `ProtectHome=false` if it lives under `/home`.

### Tool groups

Tools come in groups you tick in the Agent tab, so you can hand the model a
shell without also handing it the ability to open pull requests:

| group | tools |
| --- | --- |
| Shell | `bash` |
| Files | `read_file`, `write_file`, `edit_file`, `list_dir`, `search_files` |
| Git | `git_status`, `git_diff`, `git_log`, `git_branch`, `git_commit`, `git_checkout`, `git_push` |
| GitHub | `github_list_repos`, `github_clone`, `github_list_prs`, `github_read_pr`, `github_list_issues`, `github_read_issue` |
| GitHub write | `github_create_issue`, `github_comment`, `github_create_pr` |
| Web | `web_fetch`, `web_search` |

Unticked groups aren't offered to the model at all, and are refused
server-side if it invents a call anyway. **GitHub write is off by default** --
those actions are publicly visible, so turning them on is deliberate. The
GitHub groups only become available once a token is connected. Unticking
*every* group is allowed and means what it says: no tools, so the message is
an ordinary chat.

### Tool rounds

One message can make many tool calls; reviewing a real codebase is dozens of
reads. The budget defaults to 100 rounds, adjustable per browser in the Agent
tab (up to 500).

Running out **does not end the chat.** The model is asked once more with no
tools attached, so it reports what it found and what it was mid-way through,
and a **Continue** button appears. The tool calls and their results stay in
the conversation, so continuing resumes from the work itself rather than from
a summary of it.

Smaller models sometimes get stuck repeating one call. The third identical
call *in a row* -- same tool, same arguments, nothing in between -- is
answered with a nudge instead of being run again. Any different call in
between resets that, since a write or a checkout can genuinely change what
the repeated call would return.

### Using it

1. Pick a backend + model that supports tool/function calling.
2. Turn on **Agent tools**. Tick the directories the job needs and the groups
   it should be allowed to use.
3. Ask for something that needs the machine: *"read main.py and tell me what
   the failover logic does"*, *"clone octo/demo and summarise its layout"*,
   *"review PR #7 in octo/demo"*.
4. Watch the chat: assistant text, a collapsible card per tool call showing
   arguments and real output (failures open themselves), then the final answer.

Each message re-sends the current directories, groups and switch states, so
flipping a switch takes effect on the very next message -- there's no session
to restart.

> **The real safety boundary is the machine itself, not the sandbox switch.**
> Even sandboxed, `bash` can `cd` elsewhere or use absolute paths -- the
> confinement is a guardrail for the file tools, not a jail. Commands run as
> whatever unprivileged user the proxy runs as (`claude-proxy` under systemd).
> Run this where you'd be fine losing whatever that user can touch.

---

## GitHub and web integrations

### GitHub

Connect a personal access token (Settings → Developer settings → Personal
access tokens, `repo` scope) in the **GitHub** tab. It's stored in
`integrations.json` with `chmod 600`, is never sent to a model, never appears
in a chat transcript, and is never shown back in the UI -- the API only
reports whether one is configured.

With a token connected:

- The GitHub tab lists your repositories, and **↓** clones one into
  `AGENT_CLONE_ROOT` (default `workspace/repos`) and registers it as a working
  directory, so the file and git tools can work on it right away.
- The agent's `github_clone` tool does the same mid-conversation, adding the
  clone to the current run's scope so it can keep going without you ticking a
  box.
- Cloning authenticates with a per-command HTTP header rather than embedding
  the token in the remote URL, so **the token never lands in `.git/config`**.
  Pushes authenticate the same way.
- `GITHUB_API` defaults to `https://api.github.com`; point it at a GitHub
  Enterprise host and the clone URL follows. (It seeds the `github_api` field
  in `integrations.json`; a value saved through the API wins.)

### Web search

Paste a **Brave** or **Tavily** key in the Agent tab to enable `web_search`.
`web_fetch` works without any key.

Pick the provider that matches your key in the same form. A Tavily key sent
to Brave comes back as `422 SUBSCRIPTION_TOKEN_INVALID`, which reads like a
dead key but means the wrong provider is selected. With no key set at all,
`web_search` is dropped from the model's tool list rather than offered as a
tool that can only fail.

### Reaching the local network

`web_fetch` refuses hosts that resolve to a private, loopback or link-local
address -- **including after a redirect**, since every hop is checked before
it's fetched. That stops a URL the model picked from being used to read this
machine's own services or a cloud metadata endpoint.

Opening that up takes two deliberate steps, because it's the difference
between "the model can read the web" and "the model can read your network":

1. **The operator allows it:** set `AGENT_WEB_ALLOW_PRIVATE=1` in `.env` and
   restart. This is a ceiling, not a switch — it only decides whether the
   toggle can do anything.
2. **The message asks for it:** turn on the **LAN access** switch above the
   composer. It defaults to off and is decided per message.

With the env var unset the switch stays locked, and a request asking for LAN
access is clamped server-side rather than honoured — which matters because
the UI has no auth, so what the browser sends is a request, not a decision.

---

## Deployment

### Linux (systemd)

For a persistent service on a box/VM/LXC instead of a terminal you keep open.
`deploy/claude-proxy.service` is a template you can adapt.

```bash
# 1. Code and venv
sudo git clone https://github.com/thelorax1775/claude-proxy /opt/claude-proxy
cd /opt/claude-proxy
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. Config (both gitignored, never committed)
cp .env.example .env
cp backends.example.json backends.json
# backends.json: real provider keys/URLs
# .env: PROXY_HOST/PORT. If Claude Code runs on a DIFFERENT machine, set
# PROXY_HOST to this box's Tailscale/LAN IP -- never a public interface.

# 3. Dedicated unprivileged user (matches the unit's User=/Group=)
sudo useradd --system --no-create-home --shell /usr/sbin/nologin claude-proxy
sudo chown -R claude-proxy:claude-proxy /opt/claude-proxy

# 4. Install and start
sudo cp deploy/claude-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now claude-proxy
sudo systemctl status claude-proxy
```

Then set `ANTHROPIC_BASE_URL` to `http://<its-tailscale-ip>:8082` on whatever
machine runs `claude` (the **CLI** tab will generate this for you).

Logs: `journalctl -u claude-proxy -f`. Config changes need
`sudo systemctl restart claude-proxy` -- `.env` and `backends.json` are read
at process startup, not live-reloaded.

> A restart can take up to 90s if a browser has the UI open: uvicorn waits for
> connections to close before systemd's stop timeout kills it. Add
> `TimeoutStopSec=10` to the unit if that's annoying.

### Updating

The install above is a git clone, so:

```bash
cd /opt/claude-proxy
sudo -u claude-proxy git pull
sudo -u claude-proxy ./.venv/bin/pip install -r requirements.txt
sudo systemctl restart claude-proxy
```

`.env`, `backends.json`, `workdirs.json`, `integrations.json`, `sessions/`
and `workspace/` are all gitignored, so a pull never touches your config or
saved work.

---

## Reference

### Files and state

Everything the project needs is in this one repo: the proxy (`main.py`), the
agent layer (`agent_tools.py`, `git_tools.py`, `github_api.py`,
`web_tools.py`, `file_search.py`), saved chats (`sessions.py`), credentials
(`integrations.py`), and a static web UI. All state specific to your setup is
created locally and gitignored, so cloning this repo is safe to do publicly:

| file | holds |
| --- | --- |
| `.env` | `PROXY_HOST`/`PORT` and agent limits |
| `backends.json` | provider URLs, models, API keys |
| `workdirs.json` | directories agent mode may work in |
| `integrations.json` | GitHub token, web-search key (written `chmod 600`) |
| `sessions/` | saved chats |
| `workspace/` | the default agent working directory |

See `.env.example` for every environment variable and its default.

### Verify the translation layer directly

```bash
curl -N -X POST http://127.0.0.1:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-claude-proxy" \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "max_tokens": 512,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say hello in exactly five words."}
    ]
  }'
```

You should see Anthropic-style SSE events (`message_start`,
`content_block_delta`, ...) with the completion text streaming through. Then
`curl http://127.0.0.1:8082/status` to see which backend served it.

### Endpoints

| endpoint | purpose |
| --- | --- |
| `POST /v1/messages` | What Claude Code talks to (Anthropic → OpenAI translation) |
| `POST /v1/messages/count_tokens` | Rough `len/4` estimate for context bookkeeping |
| `GET /health` | Liveness plus the configured backend names |
| `GET /status` | Last 50 backend attempts, newest last |
| `GET /` | The web UI |

### Known limitations

- Images in messages are dropped with a placeholder, not forwarded.
- Extended thinking and prompt caching are Anthropic-specific and not
  emulated.
- Token counts from `/v1/messages/count_tokens` are a rough `len/4` estimate,
  fine for context-window bookkeeping but not exact.
- Failover only happens *before* a streaming response starts sending to
  Claude Code. If a backend dies mid-stream after already answering, that turn
  errors out rather than silently jumping to the next backend -- Claude Code
  will retry.
- Every backend model must support tool calling; see
  [Which model actually answers](#which-model-actually-answers).
