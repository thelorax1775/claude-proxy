# nim-claude-proxy

Runs Claude Code against a model of your choosing instead of Anthropic, by
translating between Anthropic's Messages API (what Claude Code speaks) and
any OpenAI-compatible chat completions API -- with automatic failover
across a list of backends.

**Where this lives:** `C:\Users\Precision-station\nim-claude-proxy\` and
pushed to `github.com/thelorax1775/nim-claude-proxy` (private). Everything
the project needs is in this one folder -- `main.py` plus a handful of
config/doc files, no external state beyond `backends.json` (gitignored,
holds real keys, never pushed). If you ever need a different AI/tool to
pick this up, clone the repo and read this README; there's nothing else to
hand over.

**What this actually is:** Claude Code stays Claude Code (the CLI/tool-use
harness). The *model* answering prompts is whichever backend in
`backends.json` actually responds. You are not getting Anthropic's Claude
models for free; you're using Claude Code as a CLI frontend for other
models.

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
- **Local Ollama (a container on your LAN)** -- no key needed, replace
  `<ct-102-ip>` in `backends.json` with its real IP, and make sure the model
  tag you list is actually pulled (`ollama pull llama3.1:8b`) and supports
  tool calling.

Model IDs in `backends.example.json` were current as of this proxy's
build date -- providers rename/rev models. If a backend keeps failing,
check its console for the current model slug before assuming something
else is broken.

You don't need every provider filled in -- fewer entries just means a
shorter failover chain. Just keep Ollama as the last entry so there's
always a final backend that can't run out.

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
box/LXC instead of a terminal you have to keep open. `deploy/nim-claude-proxy.service`
is a systemd unit template, in the same style as this homelab's other
services (e.g. `ollama.service` on ai-audit).

```bash
# 1. Get the code onto the box and set up a venv
sudo git clone https://github.com/thelorax1775/nim-claude-proxy /opt/nim-claude-proxy
cd /opt/nim-claude-proxy
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
sudo useradd --system --no-create-home --shell /usr/sbin/nologin nim-claude-proxy
sudo chown -R nim-claude-proxy:nim-claude-proxy /opt/nim-claude-proxy

# 4. Install and start the service
sudo cp deploy/nim-claude-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nim-claude-proxy
sudo systemctl status nim-claude-proxy
```

Then point `ANTHROPIC_BASE_URL` at wherever this box is reachable, e.g.
`http://<its-tailscale-ip>:8082`, from whatever machine runs `claude`.

Logs: `journalctl -u nim-claude-proxy -f`. Config changes need
`sudo systemctl restart nim-claude-proxy` to take effect (`.env` and
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
$env:ANTHROPIC_API_KEY = "local-nim-proxy"   # any non-empty placeholder
claude
```

Bash:
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=local-nim-proxy
claude
```

## Verify the translation layer directly

```bash
curl -N -X POST http://127.0.0.1:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-nim-proxy" \
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
