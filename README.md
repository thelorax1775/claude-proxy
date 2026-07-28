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

Two optional command-line tools, only for agent mode: `git` for the git and
GitHub tool groups, and an OpenSSH client (`ssh`/`scp`) for the SSH group.
If one is missing the group still ticks; the tool that needs it says so when
it runs.

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

Change the mapping in `backends.json`, or from the pencil button on the
backend's card in the **Backends** tab. A backend missing one of the two
fields falls back to the other rather than failing.

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

A backend that fails then **sits out for a while** rather than going straight
back to the front of the queue: 60 seconds after a rate limit or a connection
error, 300 after a 401/403, since a rejected key doesn't recover the way a
quota does. Without that, every request re-pays the same 429 before falling
through to an account that can actually answer. `Retry-After` is honoured when
the provider sends one, a success clears it early, and so does pasting a new
key. A benched backend shows a **cooling** tag on its card, so it never
silently vanishes from the chain. The two durations are `BACKEND_COOLDOWN` and
`BACKEND_AUTH_COOLDOWN` in `.env`.

Cooling can never empty the chain: if every backend is benched they're all
tried anyway. An answer from an over-quota account beats no answer.

Where a provider will say how much allowance is left, the chain is **ordered by
what's left rather than tried top to bottom**: whichever backend has the most
of its weekly quota remaining goes first, and one that has spent its quota goes
last. Without this a pool of accounts drains one at a time — the first is
flattened while the rest sit idle, and once it's dead every single request pays
a 429 before the chain moves on.

Two things keep it from being disruptive:

- **Ten-point bands.** The figures are rounded and move constantly, and the
  front of the chain is the end that gets used, so ordering on the raw
  percentage would change the leader every couple of minutes and never settle.
  A backend only overtakes another once it's a band emptier; inside a band your
  configured order stands.
- **Only measured backends move.** A backend with no usage API — or one whose
  usage lookup just failed — keeps its exact position, and the measured ones
  are shuffled between the slots they already occupied. So a paid endpoint you
  deliberately put first stays first, and a usage API outage falls back to the
  configured order rather than to a guess.

Demotion of a spent account is a reorder and not a filter: it's still tried if
everything above it fails, because a rounded figure and a window that may have
just rolled over are not proof it will refuse. `/status` reports the effective
`chain`, what's currently `demoted`, and the `order` mode; `BACKEND_ORDER=configured`
uses `backends.json` exactly as written. See
[Remaining allowance](#remaining-allowance).

### Remaining allowance

A free-tier key that has run out doesn't announce itself: it 429s, gets
benched, and the chain quietly gets shorter. Where the provider has an API for
it, each backend card shows a meter for how much of its allowance is gone, and
`GET /api/usage` returns the same thing as JSON.

**Ollama Cloud** is the provider implemented. `GET https://ollama.com/api/usage`
reports `limits.weekly.usage` as a *fraction of the weekly allowance already
spent*, so `0.957` means 4.3% left. Anything else reports
`supported: false` rather than guessing — adding a provider is a fetcher and
a parser in `usage.py`.

Answers are cached for two minutes and fetched concurrently; **Refresh usage**
on the Backends tab forces a re-read, and changing a key or base URL drops that
backend's cached figures, since a different key is a different account. None of
it is on the request path: routing reads whatever happens to be cached and
never waits on a provider, so a usage API being down can slow nothing and break
nothing. The cache is kept warm by traffic rather than a timer — a busy proxy
re-checks every couple of minutes, an idle one never does.

The meter is one hue for magnitude with `--danger` reserved for the state
change at 100%, and the number beside it always states the figure. That's
deliberate: a red/amber/green scale built from this UI's own tokens fails
colourblind separation (`warn`↔`ok` is ΔE 5.5 under protanopia), so the colour
agrees with the label rather than carrying it.

A conversation that no longer fits a model's context window also falls through
to the next backend, since window sizes differ -- see
[When a conversation gets too long](#when-a-conversation-gets-too-long).

Watch it happen live in the **Backends** tab, or:

```bash
curl http://127.0.0.1:8082/status
```

The **Backends** tab is the easiest way to manage this: provider presets, a
**Detect models** button that pulls the real model list from the provider
instead of you typing model IDs by hand, and drag-free reordering. It writes
`backends.json` for you, so keys never go through a chat transcript.

**To change an existing backend**, press the pencil on its card: name, base
URL, both models and the key. **Leave the key field blank to keep the saved
one** -- the API never hands a key back to the browser, so a blank field means
"unchanged", not "erase". Detect models works here too; the server uses the
stored key, which the browser doesn't have. Renaming keeps the key, so there's
no reason to delete and re-add.

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

### When a conversation gets too long

Every model has a context window, and a long chat -- especially an agent run,
where each tool result is appended verbatim -- eventually exceeds it. The
provider answers `400`, which is otherwise the shape of "your request is
malformed": not worth retrying, because the next backend would say the same.

Context is the exception, so it gets treated as its own thing:

- **It fails over.** Backends differ in context window as much as in anything
  else, and a conversation too long for an 8k model is ordinary for a 128k
  one. Put a big-window backend in the chain and long chats simply carry on
  there. The attempt shows as `context overflow` in the Backends tab.
- **It doesn't bench the backend.** Unlike a rate limit, this says nothing
  about the account -- the next message may well be short, so the backend
  stays in the chain.
- **If nothing will take it, the chat still isn't stuck.** You get an offer to
  **drop the older half and retry** rather than a provider's raw complaint.
  That trims what the *model* sees, always on a whole-exchange boundary so a
  tool result never gets separated from the call it answers. The transcript on
  screen keeps everything, and nothing is written to `sessions/` that wasn't
  already there.

For Claude Code the proxy passes the provider's own context error through
(`invalid_request_error`, "prompt is too long") rather than a generic failure,
which is what Claude Code reads to decide to compact and retry by itself.

### No auth, by design

There is no login and no shared key: **anything that can reach the port can
spend your API keys, and can switch on agent mode and get a shell on the
machine running the proxy.** `PROXY_HOST` is the only thing limiting that.
Keep it on `127.0.0.1`, or on a private interface (e.g. a Tailscale IP) that
only you can reach. Never expose this port to the internet.

---

## Agent tools

Lets the model in the web UI's chat actually work on the machine hosting the
proxy: run commands, read and edit files, drive git, use GitHub, fetch from
the web, and reach other machines over SSH. This is **separate from Claude
Code**, which has its own tools and is unaffected by any of it. Everything is
controlled from the UI -- no env var to set first, no restart.

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

### Approving tool calls

The dropdown to the right of the switches decides whether you see a tool call
before it happens:

| mode | what happens |
| --- | --- |
| **Manually approve** | The run stops before each call and shows the tool and its exact arguments. Nothing executes until you answer. |
| **Automatically approve** | Calls run as they come. **Stop** is the only brake. |
| **Skip all approvals** | The same, stated plainly. This is the default and how every earlier version behaved. |

An approval card offers **Approve**, **Deny** and **Approve all remaining**.
That last one matters: a code review can be a hundred tool rounds, and
approving each `read_file` individually is a mode nobody would leave switched
on. It applies to the rest of that one message only -- the next message asks
again.

Denying doesn't derail the run. The model is told the call was declined and
asked to carry on without it or explain what it needed, so it can adapt
rather than retry in a loop. Nobody answering within
`AGENT_APPROVAL_TIMEOUT` (default 600s) counts as a denial.

The run is genuinely blocked server-side while a card is on screen: the loop
is parked on a future, and the decision arrives on a separate request
(`POST /api/approvals/{id}`), because a response that is mid-stream can't read
from its own request. While it waits, the stream emits SSE comment keepalives
so an idle-connection timeout somewhere in the middle can't kill a run that is
simply waiting on a person.

> Manual mode is per browser and remembered. It is **not** a security
> boundary: the proxy has no auth, so anyone who can reach the port can also
> just switch it to Skip. It stops the model surprising *you*, not somebody
> else deciding to.

### The + menu

Next to the permission dropdown, for the things you'd otherwise open the side
panel for mid-conversation:

- **Add files** (`Ctrl+U`, or drag anything onto the chat) uploads into the
  first ticked working directory and writes the paths into the composer, so
  you can say what to do with them. 25 MB per file.
- **Working directories** and **Tool groups** are the same tick-boxes as the
  Agent tab, sharing the same state -- changing one updates the other.
- **Clone a repository** takes `owner/repo` and registers the clone as a
  working directory without a trip to the GitHub tab.

### Working directories

The Agent tab lists every directory agent mode may work in. `workspace/`
(from `AGENT_WORKSPACE`, created automatically) is always there and can't be
removed; add any other directory on the proxy's machine by typing its path.
Tick the ones the current session should see -- the first ticked directory is
the primary one, shown as `primary` on its card and used as `bash`'s working
directory.

Added directories are stored in `workdirs.json` (gitignored) and survive a
restart. Ticking is per browser, not shared.

> A directory added here is writable straight away. The unit no longer sets
> `ProtectSystem`/`ReadWritePaths`, so there is no second list to keep in sync
> and no `daemon-reload` after adding one.

### Tool groups

Tools come in groups you tick in the Agent tab, so you can hand the model a
shell without also handing it the ability to open pull requests:

| group | n | tools |
| --- | --- | --- |
| Shell | 1 | `bash` |
| Files | 6 | `read_file`, `write_file`, `edit_file`, `list_dir`, `search_files`, `find_files` |
| Files: move & delete | 2 | `move_file`, `delete_file` |
| Git | 12 | `git_status`, `git_diff`, `git_log`, `git_branch`, `git_commit`, `git_checkout`, `git_push`, `git_fetch`, `git_pull`, `git_show`, `git_stash`, `git_restore` |
| GitHub | 11 | `github_list_repos`, `github_clone`, `github_list_prs`, `github_read_pr`, `github_list_issues`, `github_read_issue`, `github_ci`, `github_read_file`, `github_list_branches`, `github_search_code`, `github_read_commit` |
| GitHub write | 7 | `github_create_issue`, `github_comment`, `github_create_pr`, `github_merge_pr`, `github_review_pr`, `github_close_issue`, `github_write_file` |
| Web | 3 | `web_fetch`, `web_search`, `http_request` |
| SSH | 4 | `ssh_list_hosts`, `ssh_run`, `ssh_upload`, `ssh_download` |

Unticked groups aren't offered to the model at all, and are refused
server-side if it invents a call anyway. **The tick is the only gate.** No
group is ever greyed out or withheld: tick GitHub before connecting a token,
or SSH before registering a host, and the tools will tell you what's missing
when they run -- "no GitHub token configured -- add one in the UI's Agent tab"
is a more useful answer than a disabled checkbox, and it doesn't disable the
control you came to use while you set the credential up. Unticking *every*
group is allowed and means what it says: no tools, so the message is an
ordinary chat.

**Three groups are off by default**, because they do things you can't take
back:

- **Files: move & delete** -- `delete_file` removes one file at a time and
  refuses directories outright, so there's no recursive tree delete to get
  wrong. Read and edit live in the ordinary Files group, so you can hand over
  editing without handing over deletion.
- **GitHub write** -- publicly visible actions. `github_merge_pr` and
  `github_review_pr` are the two worth thinking about before ticking: an
  approval can be what allows a merge.
- **SSH** -- `ssh_run` is a shell on a *different* machine, one this proxy's
  sandbox has no authority over at all. See [SSH hosts](#ssh-hosts).

Things worth knowing about individual tools:

- `find_files` matches file *names*; `search_files` matches file *contents*.
  Both skip `.git`, `node_modules`, `.venv` and similar.
- `read_file` takes `offset`/`limit` line numbers, so a file bigger than the
  output budget can be paged through rather than silently truncated.
- `git_pull` is **fast-forward only**. If the branches have diverged it says
  so instead of starting a merge or rebase nobody is watching.
- `git_restore` refuses to run without an explicit filename -- "restore
  everything" would discard uncommitted work with no undo.
- `github_read_file` and `github_read_commit` work without cloning, which is
  much faster when you only want an opinion on one file.
- `github_search_code` uses GitHub's search API, which **does not reliably
  index private repositories**. On a private repo it returns no matches even
  when the text is there; clone it and use `search_files` instead.
- `http_request` is for APIs -- explicit method, headers and body, response
  returned raw. `web_fetch` is for reading pages and strips markup. Both run
  through the same redirect-checked destination guard, so both respect the
  LAN switch.
- `bash` takes an optional `timeout` (default 30s, up to 300). Raise it for a
  build or a test suite rather than watching the default cut one short.

### SSH hosts

Lets agent mode reach machines *other* than the one running the proxy: run
commands on them, and copy files each way.

**The registry is the allowlist.** The tools take a registered **name** and
there is no way to hand them a connection string, so the set of machines the
model can reach is exactly the set in the registry. Hosts live in
`ssh_hosts.json` (gitignored, written `chmod 600`).

Add them in the Agent tab, or let the model add one with `ssh_add_host` when
you give it the details in chat — "the NAS is at 10.0.0.5, log in as admin"
is enough. **`ssh_add_host` has no password parameter**, and that's the point
rather than an omission: a password typed into chat is in the prompt, in the
stored conversation, and in every request sent upstream from then on — it has
left the machine before it is ever used. A path is just a path, so a key can
come from chat safely.

So a host that needs a password is registered without one, and the tool tells
the model to send you to the **key icon** on that host's card, where you type
it into the browser. It goes straight to `ssh_hosts.json` without passing
through a model, a prompt, or a transcript. The model is also told, in its
system prompt, never to ask for one in chat.

**Auth is a key or a password**, per host. Leave **Key file** blank to use the
proxy user's default keys or agent, or give an absolute path to a specific one.

Prefer a key. Use a **Password** for the machines that will never have one — a
router, an appliance, a NAS, somebody else's box — where "install a key" isn't
advice you can act on. The tradeoff is real and worth stating plainly: the
password is stored **in plain text** in `ssh_hosts.json`. The file is `chmod
600` and gitignored, and the password is never returned by the API, shown in
the UI, or visible to the model — but anyone who can read that file, or a
backup of it, has the password.

Whichever you use, ssh must never sit at a prompt: there's nobody at a
terminal, so a prompt isn't a question, it's a tool call that hangs until the
timeout with nothing to explain why. Without a password that means refusing to
prompt at all (`BatchMode=yes`, `PasswordAuthentication=no`,
`NumberOfPasswordPrompts=0`). With one it means the opposite — prompting has to
be possible for the password to be delivered — so the guarantee is
re-established as *exactly one* attempt, restricted to the methods a password
can actually satisfy (`NumberOfPasswordPrompts=1`,
`PreferredAuthentications=password,keyboard-interactive`). A wrong password
comes back as a permission error in a couple of seconds rather than three
retries and a wait.

The password reaches ssh through `SSH_ASKPASS`, answered by `ssh_askpass.py`.
That keeps the system `ssh` and `scp` binaries doing all the work — no
`sshpass` to install, no second SSH implementation to keep in step — and keeps
the password out of `argv`, which every process on the machine can read,
putting it in the environment, which only this user and root can. The helper
answers password prompts only; ssh uses the same mechanism to ask whether to
accept an unrecognised host key, and a helper that answered that would hand the
password to whoever picked up the connection. Requires OpenSSH 8.4+ on the
proxy's machine for `SSH_ASKPASS_REQUIRE`.

A password does not relax any of this. If anything it raises the stakes: a key
is never sent to the far end and a password is, so a substituted host key with
password auth hands over the credential itself.

**Host keys use `StrictHostKeyChecking=accept-new`**: the first connection
records the host's key, and a key that *changes* afterwards is refused
loudly. `no` would silently accept a substituted key -- which is to say,
accept a machine-in-the-middle -- and `yes` makes first contact fail until
someone hand-edits `known_hosts`. Set `AGENT_SSH_STRICT_HOST_KEY=yes` if you
pre-populate `known_hosts` and want the stricter one.

Each host card has a **Test** button that connects and runs `true`. Use it
before involving a model: it turns "the tool timed out" into
`Permission denied (publickey)` or `Connection timed out`, which is the
difference between a five-minute fix and an afternoon.

Worth knowing:

- **The sandbox does not extend to the far side.** Working directories bound
  the *local* half of `ssh_upload`/`ssh_download`, and nothing else. On the
  remote machine the model has whatever that account has, with no undo.
- Sessions are non-interactive, so anything that waits for input times out
  instead of prompting -- prefer `-y` and friends.
- `ssh_run` takes an optional `timeout` (default 120s, up to 600) because a
  remote build or package update routinely needs more than a local command.
  A command that times out is disconnected, but **may keep running on the
  remote host**: ssh has no way to kill it from this end.
- `ssh_download` refuses to overwrite an existing local file.
- Keys come from the service user's `~/.ssh` in the ordinary way, unless a
  host card names its own **Key file**. `known_hosts` defaults to
  `~/.ssh/known_hosts` and the directory is created if it doesn't exist --
  a service account often has no home on disk, and ssh will not record a host
  key it cannot write.

> Nothing in the unit stands between the tools and `~/.ssh` any more — the
> filesystem protections that used to hide it are gone, and the process has
> the reach of the `claude-proxy` user. Generate a key for that account,
> install the `.pub` on the machines you want to reach, and the group works:
>
> ```sh
> sudo -u claude-proxy ssh-keygen -t ed25519 -N ''
> ```
>
> The account and the machine are the boundary now, so run this somewhere
> you're comfortable with that — a container or a VM, not a workstation with
> your own files on it.

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

The budget counts *rounds*, and one round can ask for several tools at once,
so there's a second cap of 25 calls per round. Anything past it is answered
with an error rather than run, which keeps one confused answer from spending
the whole budget before the loop gets to look at anything.

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
`ssh_tools.py`, `web_tools.py`, `file_search.py`), saved chats
(`sessions.py`), credentials (`integrations.py`), tests (`tests/`), and a
static web UI. All state specific to your setup is created locally and
gitignored, so cloning this repo is safe to do publicly:

| file | holds |
| --- | --- |
| `.env` | `PROXY_HOST`/`PORT` and agent limits |
| `backends.json` | provider URLs, models, API keys |
| `workdirs.json` | directories agent mode may work in |
| `integrations.json` | GitHub token, web-search key (written `chmod 600`) |
| `ssh_hosts.json` | SSH hosts agent mode may connect to (written `chmod 600`) |
| `sessions/` | saved chats |
| `workspace/` | the default agent working directory |

Every stored secret -- the GitHub token, the search key, and each provider key
in `backends.json` -- is blanked out of tool output before it reaches a chat
bubble, a log line or a saved session. So asking the agent to read
`backends.json`, or to run `env`, returns `***` rather than your keys.

See `.env.example` for every environment variable and its default.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

They cover the Anthropic↔OpenAI translation, the streaming SSE conversion,
failover and cooldown, the backend API, the agent loop's conversation
bookkeeping, and the SSH registry's validation. Everything runs against
in-process stubs, so the suite needs no network, no API key and no `ssh`
binary, and it redirects every state path to a temp directory -- running it
can't touch your real `backends.json` or `sessions/`.

`requirements.txt` is unchanged: the proxy itself still needs only four
packages. The test-only dependencies live in `requirements-dev.txt`.

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
| `GET /status` | Last 50 backend attempts, newest last, plus what's cooling, the effective chain order, and what's demoted |
| `GET /api/usage` | Remaining allowance per backend; `?refresh=1` bypasses the cache |
| `POST /api/approvals/{id}` | Answer a tool call a manual-mode run is parked on |
| `POST /api/upload?name=&path=` | Raw-body file upload into a registered working directory |
| `GET/POST /api/ssh-hosts` | List and register SSH hosts |
| `POST /api/ssh-hosts/{name}/password` | Set or clear one host's password; blank body clears |
| `DELETE /api/ssh-hosts?name=` | Unregister one |
| `POST /api/ssh-hosts/{name}/test` | Connect and run `true`; reports latency or the ssh error |
| `GET /` | The web UI |

`PATCH /api/backends/{name}` takes `name`, `base_url`, `big_model`,
`small_model` and `api_key`. A blank `api_key` leaves the stored key alone;
send `null` to clear it. `POST /api/probe-models` accepts either a
`base_url` + `api_key` pair (the add form) or the `name` of a saved backend,
in which case it uses that backend's stored key.

### Known limitations

- Images in messages are dropped with a placeholder, not forwarded.
- Extended thinking and prompt caching are Anthropic-specific and not
  emulated.
- Token counts from `/v1/messages/count_tokens` are a rough `len/4` estimate
  over the *text* of a request only: tool schemas and tool results aren't
  counted, so an agentic turn reads as much smaller than it is.
- `output_tokens` on a streamed response is the provider's own count when it
  volunteers one, and the same `len/4` estimate otherwise. `input_tokens` on
  `message_start` is still reported as 0.
- Trimming an over-long conversation is manual and blunt -- a button that
  drops the older half. There's no summarising or token-aware trimming, so
  what it drops is gone from the model's view for the rest of the chat.
- Failover only happens *before* a streaming response starts sending to
  Claude Code. If a backend dies mid-stream after already answering, that turn
  errors out rather than silently jumping to the next backend -- Claude Code
  will retry.
- Every backend model must support tool calling; see
  [Which model actually answers](#which-model-actually-answers).
