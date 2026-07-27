# claude-proxy

Runs Claude Code against a model of your choosing instead of Anthropic, by
translating between Anthropic's Messages API (what Claude Code speaks) and
any OpenAI-compatible chat completions API -- with automatic failover
across a list of backends.

Everything the project needs is in this one repo -- `main.py` plus a
handful of config/doc files. All state that's specific to your setup
(API keys, which backends you use) lives in `backends.json`, which you
create locally and which is gitignored -- it never gets committed, so
cloning this repo is safe to do publicly.

**What this actually is:** Claude Code stays Claude Code (the CLI/tool-use
harness). The *model* answering prompts is whichever backend in
`backends.json` actually responds. You are not getting Anthropic's Claude
models for free; you're using Claude Code as a CLI frontend for other
models.

## Web UI

Once the proxy is running, open `http://127.0.0.1:8082/` (or wherever you
bound it) in a browser for:

- **Backend management** -- pick a provider preset (or "Custom"), paste in
  its API key, hit "Detect models" to pull the real model list from that
  provider instead of typing model IDs by hand, and save. Reorder or
  remove backends with the arrows/× on each card. This edits `backends.json`
  for you -- no need to hand-edit JSON or paste keys into a chat transcript.
- **A test chat** -- pick any configured backend + model from the dropdowns
  and talk to it directly, with streaming responses (no Anthropic
  translation -- this is a human talking to a provider, separate from how
  Claude Code uses the proxy via `/v1/messages`). If that backend gets
  rate-limited or its key dies mid-conversation, the chat transparently
  falls through the rest of your configured backends (same conversation
  history, no error shown) and a small system note announces the switch --
  the dropdowns update to match, so your next message starts there directly.
- **Agent tools** -- lets the model in the test chat actually run commands
  and touch files on the container, off by default with a server-side kill
  switch. See "Agent tools" below for how to turn it on and use it.

Everything here runs API keys through this local server only -- keys never
leave the machine the proxy runs on except to reach the provider you picked.

## Prerequisites

- [Claude Code](https://github.com/anthropics/claude-code) installed (`npm install -g @anthropic-ai/claude-code`)
- Python 3.10+
- At least one place for requests to actually go: an account with a free-tier
  OpenAI-compatible API (Groq, Google AI Studio, Cerebras, OpenRouter,
  NVIDIA NIM -- see below), and/or your own running Ollama instance

```bash
git clone https://github.com/thelorax1775/claude-proxy
cd claude-proxy
```

## How the failover works

`backends.json` is an ordered list. On every request, the proxy tries
backend #1; if it gets rate-limited (429), errors (5xx), or its key is
dead (401/403), it logs that and tries backend #2, and so on, down to your
local Ollama instance last (which has no quota to exhaust). First backend
that answers wins. Watch it happen live:

```bash
curl http://127.0.0.1:8082/status
```

**Important, read once:** sign up for one account each at *different*
services (Groq, Google AI Studio, Cerebras, OpenRouter, NVIDIA NIM below),
not multiple accounts on the *same* service. Free tiers are one-per-person
by design, and creating several accounts to multiply a single provider's
quota is against most of their terms of service -- it risks getting all of
those keys banned together, which breaks the whole chain at once. Five
different providers gets you the same resilience with no such risk.

## Setup

```bash
pip install -r requirements.txt
cp backends.example.json backends.json
# edit backends.json: paste your real API key for each provider you signed up for
```

Where to get each free key:
- **Groq** -- console.groq.com -> API Keys. Generous free tier, fast, good tool-calling support.
- **Google AI Studio (Gemini)** -- aistudio.google.com -> Get API key. Free tier, OpenAI-compatible endpoint.
- **Cerebras** -- cloud.cerebras.ai -> API Keys. Free tier.
- **OpenRouter** -- openrouter.ai/keys. The `:free` suffixed models cost nothing (rate-limited).
- **NVIDIA NIM** -- build.nvidia.com -> API Keys. Free trial credits, not unlimited.
- **Local Ollama** -- if you're running Ollama yourself, no key needed.
  Point `base_url` at wherever it's reachable (e.g. `http://localhost:11434/v1`
  for raw Ollama, or through a gateway like Open WebUI's own API if you've
  locked Ollama down to localhost -- see "A note on Open WebUI" below). Make
  sure the model tag you list is actually pulled (`ollama pull llama3.1:8b`)
  and supports tool calling.

Model IDs in `backends.example.json` were current as of this proxy's
build date -- providers rename/rev models. If a backend keeps failing,
check its console for the current model slug before assuming something
else is broken.

You don't need every provider filled in -- fewer entries just means a
shorter failover chain. Just keep Ollama as the last entry so there's
always a final backend that can't run out.

### A note on Open WebUI

If Ollama is only reachable through Open WebUI (e.g. you've deliberately
bound Ollama to `127.0.0.1` and Open WebUI is the only exposed gateway),
point `base_url` at Open WebUI instead: `http://<host>:8080/api`, with an
`Authorization: Bearer <key>` generated in Open WebUI's Settings > Account >
API Keys. Its OpenAI-compatible route is `POST /api/chat/completions`, not
`/v1/chat/completions` -- that path just serves Open WebUI's own frontend
on most versions. This proxy's default `{base_url}/chat/completions` URL
construction handles both cases correctly as long as `base_url` includes
whichever prefix (`/api` or nothing) your setup actually needs.

## Run the proxy

PowerShell:
```powershell
./start.ps1
```

Bash (also what the systemd unit below calls):
```bash
chmod +x start.sh
./start.sh
```

Binds to `PROXY_HOST:PROXY_PORT` (default `127.0.0.1:8082`) -- see
`.env.example`. Stays localhost-only unless you deliberately widen it,
since this proxy holds multiple API keys and has no auth by default.

## Linux deployment (systemd)

Use this if you want the proxy running as a persistent service on a Linux
box/VM/LXC instead of a terminal you have to keep open. `deploy/claude-proxy.service`
is a systemd unit template you can adapt.

```bash
# 1. Get the code onto the box and set up a venv
sudo git clone https://github.com/thelorax1775/claude-proxy /opt/claude-proxy
cd /opt/claude-proxy
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. Config (both gitignored, never committed)
cp .env.example .env
cp backends.example.json backends.json
# edit both: real backend keys/URLs in backends.json; PROXY_HOST/PORT and
# PROXY_SHARED_KEY in .env -- if Claude Code runs on a DIFFERENT machine
# than this box, set PROXY_HOST to 0.0.0.0 (or this box's Tailscale IP) and
# set PROXY_SHARED_KEY, since anything beyond 127.0.0.1 has no auth otherwise

# 3. Dedicated unprivileged user (matches the unit's User=/Group=)
sudo useradd --system --no-create-home --shell /usr/sbin/nologin claude-proxy
sudo chown -R claude-proxy:claude-proxy /opt/claude-proxy

# 4. Install and start the service
sudo cp deploy/claude-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now claude-proxy
sudo systemctl status claude-proxy
```

Then point `ANTHROPIC_BASE_URL` at wherever this box is reachable, e.g.
`http://<its-tailscale-ip>:8082`, from whatever machine runs `claude`.

Logs: `journalctl -u claude-proxy -f`. Config changes need
`sudo systemctl restart claude-proxy` to take effect (`.env` and
`backends.json` are both read at process startup, not live-reloaded).

Check it's alive:
```bash
curl http://127.0.0.1:8082/health
```

## Point Claude Code at it

In the terminal you'll run `claude` from:

PowerShell:
```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8082"
$env:ANTHROPIC_API_KEY = "local-claude-proxy"   # any non-empty placeholder
claude
```

Bash:
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=local-claude-proxy
claude
```

## Verify the translation layer directly

```bash
curl -N -X POST http://127.0.0.1:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-claude-proxy" \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "max_tokens": 512,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Show Python code to convert CSV to JSON with pandas, with error handling for missing columns."}
    ]
  }'
```

You should see Anthropic-style SSE events (`message_start`,
`content_block_delta`, ...) with the actual completion text streaming
through. Then check `curl http://127.0.0.1:8082/status` to see which
backend actually served it.

## Model mapping

Claude Code sends whatever model string is active (its default, or
whatever you set `ANTHROPIC_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` to). Per
backend, the proxy buckets on the string:

- contains `haiku` -> that backend's `small_model`
- everything else (sonnet, opus, default) -> that backend's `big_model`

Change the mapping in `backends.json`, not in code.

## Agent tools

Lets the model in the web UI's test chat actually run shell commands and
read/write files on the container -- separate from Claude Code, which
already has its own tools and is unaffected by any of this. Off by
default at two levels, both of which need to be true for it to run:

1. **Server-side kill switch** -- `AGENT_TOOLS_ENABLED` in `.env`, default
   `false`. This is the real gate: if it's off, the proxy rejects any
   request asking for agent mode, full stop, regardless of what the
   browser sends.
2. **Per-session checkbox** -- "Agent tools" above the chat composer in
   the web UI, unchecked on every page load (not saved to localStorage).
   If the server-side switch is off, this checkbox shows disabled with an
   explanation on hover; checking it does nothing until an operator turns
   on the switch below.

### Turning it on

```bash
# in .env, on whichever machine/container runs the proxy:
AGENT_TOOLS_ENABLED=true
```

Then restart the proxy (`sudo systemctl restart claude-proxy` for a
systemd deployment, or just re-run `start.sh`/`start.ps1` locally) --
`.env` is only read at startup.

### Using it

1. Open the web UI, pick a backend + model that supports tool/function
   calling (same requirement as everywhere else in this proxy).
2. Check "Agent tools" above the composer.
3. Ask for something that needs the container, e.g. "list the files in
   your workspace" or "write a file called notes.txt with today's date in
   it, then read it back to confirm."
4. Watch the chat: assistant text, then a distinct bubble per tool call
   showing the command/args and its real output, then the model's final
   answer once it stops calling tools (capped at 8 tool-call rounds per
   message before the proxy gives up and reports it wasn't able to finish).

Four tools are available: `bash`, `read_file`, `write_file`, `list_dir`.
All file/directory paths are confined to a `workspace/` directory (set by
`AGENT_WORKSPACE` in `.env`, created automatically) -- attempts to read or
write outside it are rejected. `bash` runs with that directory as its
working directory, but a shell can still `cd` elsewhere or use absolute
paths; the workspace confinement is a convenience for file tools, not a
hard sandbox.

**The real safety boundary is the container itself, not the workspace
directory.** Commands run as whatever unprivileged system user the proxy
process runs as (`claude-proxy` in the systemd deployment) -- there's no
additional sandboxing beyond that. Only turn `AGENT_TOOLS_ENABLED` on for
a deployment you'd be fine losing entirely; don't enable it on a container
that holds anything you care about.

## The one thing that will actually break things

Claude Code's file editing, bash execution, etc. all run through Anthropic
tool-use (`tools` + `tool_use`/`tool_result` blocks), translated here into
OpenAI `tools`/`tool_calls`. **Every model you list must support
function/tool calling.** If one doesn't, that backend won't error loudly --
Claude Code will just seem to "forget" how to edit files or run commands
whenever failover lands on it. Check each provider's docs for
function-calling support before adding a model.

## Known limitations

- Images in messages are dropped with a placeholder, not forwarded.
- Extended thinking / prompt caching are Anthropic-specific and not emulated.
- Token counts from `/v1/messages/count_tokens` are a rough `len/4` estimate,
  not an exact tokenizer match -- fine for context-window bookkeeping, not
  exact.
- Failover only happens *before* a streaming response starts sending to
  Claude Code (the proxy checks the backend's status first). If a backend
  dies mid-stream after already answering, that turn errors out rather than
  silently jumping to the next backend -- Claude Code will just retry.
