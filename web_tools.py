"""Web access for agent mode: fetch a page as readable text, run a search.

Two leaf helpers, no project imports and no parsing dependencies -- the
proxy is meant to stay a `pip install -r requirements.txt` away from
running, so HTML is reduced with a handful of regexes rather than a real
parser. Everything here is best-effort: these run inside a tool call, so a
dead host or a PDF is a string the model can read and react to, never an
exception that kills the turn.
"""

import asyncio
import html
import ipaddress
import os
import re
import socket

import httpx

OUTPUT_LIMIT = 8000

# Fetches are capped well below any timeout the caller might have; a page
# that hasn't answered in 20s isn't worth blocking an agent turn on.
TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# Some sites serve a challenge page (or nothing) to obvious bots, and the
# point of this tool is to see what a person would see.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Read a bounded prefix of the body and stop pulling once it's reached: a
# multi-gigabyte URL must not become a multi-gigabyte string in the proxy,
# and truncating before the regex pass keeps a 50 MB page from turning into
# 50 MB of backtracking.
MAX_BODY_BYTES = 2_000_000

# web_fetch runs on behalf of a model, against a URL the model chose, from
# inside whatever network the proxy sits in -- which is the shape of an SSRF.
# Private, loopback and link-local destinations (169.254.169.254 and friends)
# are refused unless the operator opts back in. Redirects are followed by
# hand so a public URL can't bounce into one.
ALLOW_PRIVATE = os.getenv("AGENT_WEB_ALLOW_PRIVATE", "").strip().lower() in ("1", "true", "yes")
MAX_REDIRECTS = 5

SEARCH_PROVIDERS = ("brave", "tavily")
SNIPPET_LIMIT = 300

SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")

# Elements whose *contents* are markup or styling, not prose. Comments go
# first so a commented-out <script> can't leave a stray closing tag behind.
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
# <title> is in here because it's reported on the header line instead.
DROP_RE = re.compile(r"<(script|style|noscript|svg|template|title)\b[^>]*>.*?</\1\s*>", re.S | re.I)
# Same elements, opened but never closed: everything after them is source,
# not prose. The lookbehind keeps a self-closing <svg/> from eating the page.
DROP_OPEN_RE = re.compile(r"<(script|style|noscript|svg|template)\b[^>]*(?<!/)>.*", re.S | re.I)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
# Block boundaries that a reader perceives as a line break.
BREAK_RE = re.compile(r"<br\s*/?>|</p\s*>|</div\s*>|</li\s*>|</h[1-6]\s*>|</tr\s*>", re.I)
# Any remaining tag. Quoted attribute values are matched explicitly because
# they may legally contain '>' (title="a > b"), and a naive <[^>]*> would cut
# the tag short and spill its attributes into the text. Requiring a letter
# after '<' leaves unescaped prose like "5 < 6" alone.
TAG_RE = re.compile(r"""</?[!?]?[A-Za-z][^>"']*(?:(?:"[^"]*"|'[^']*')[^>"']*)*>""", re.S)
BLANK_RUN_RE = re.compile(r"\n{3,}")
# Horizontal whitespace, including the &nbsp; and zero-width space that
# html.unescape turns into literal characters.
SPACE_RUN_RE = re.compile("[ \t\u00a0\u200b\u2009\ufeff]+")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text)} bytes total]"


def html_to_text(markup: str) -> str:
    """Strip HTML down to something worth putting in a context window."""
    text = COMMENT_RE.sub(" ", markup)
    text = DROP_RE.sub(" ", text)
    # An unclosed <script>/<style> would otherwise dump its source as prose.
    text = DROP_OPEN_RE.sub(" ", text)
    text = BREAK_RE.sub("\n", text)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)

    # Source indentation and wrapping aren't content, so squeeze horizontal
    # runs per line; that also turns whitespace-only lines into true blanks,
    # which is what lets the blank-run collapse below actually fire.
    lines = [SPACE_RUN_RE.sub(" ", line).strip() for line in text.splitlines()]
    return BLANK_RUN_RE.sub("\n\n", "\n".join(lines)).strip()


def _page_title(markup: str) -> str:
    match = TITLE_RE.search(markup)
    if not match:
        return ""
    return SPACE_RUN_RE.sub(" ", html.unescape(match.group(1))).strip()


def _normalize_url(url: str) -> tuple[str, str]:
    """Returns (url, error). Bare hosts are assumed to be https."""
    url = (url or "").strip()
    if not url:
        return "", "[error] no url given"
    if url.startswith("//"):
        return "https:" + url, ""            # protocol-relative, as copied out of a page
    match = SCHEME_RE.match(url)
    if match is None:
        return "https://" + url, ""          # "example.com/x"
    if match.group(1).lower() in ("http", "https"):
        return url, ""
    if url[match.end():].split("/", 1)[0].isdigit():
        return "https://" + url, ""          # "example.com:8080/x" -- a port, not a scheme
    return "", "[error] only http(s) URLs are supported"


def _blocked_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified)


async def _destination_error(url: httpx.URL) -> str:
    """Empty string if this URL may be fetched, an [error] line if not. Every
    address the host resolves to is checked, so a name that answers with both
    a public and a private address is still refused."""
    if ALLOW_PRIVATE:
        return ""
    host = url.host
    if not host:
        return f"[error] no host in {url}"
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, url.port or (443 if url.scheme == "https" else 80),
                                        0, socket.SOCK_STREAM)
    except socket.gaierror as e:
        return f"[error] could not resolve {host}: {e}"
    except OSError as e:
        return f"[error] could not resolve {host}: {e}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _blocked_address(ip):
            return (f"[error] refusing to fetch {host}: it resolves to {ip}, which is on this "
                    "machine's own network. Set AGENT_WEB_ALLOW_PRIVATE=1 on the proxy if that "
                    "is genuinely what you want it to reach.")
    return ""


async def _read_capped(resp) -> bytes:
    """Pull at most MAX_BODY_BYTES off the wire, then stop -- the point is to
    never hold a body bigger than that, so the cap has to apply while
    downloading rather than after."""
    chunks, total = [], 0
    async for chunk in resp.aiter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_BODY_BYTES:
            break
    return b"".join(chunks)[:MAX_BODY_BYTES]


async def web_fetch(url: str, max_chars: int = OUTPUT_LIMIT) -> str:
    """Fetch a URL and return its readable text, prefixed with a source line."""
    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError):
        max_chars = OUTPUT_LIMIT
    if max_chars <= 0:
        max_chars = OUTPUT_LIMIT

    normalized, error = _normalize_url(url)
    if error:
        return error

    try:
        target = httpx.URL(normalized)
    except httpx.InvalidURL as e:
        return f"[error] {normalized} is not a usable url: {e}"

    resp = None
    try:
        # follow_redirects=False so every hop is checked before it's fetched;
        # otherwise a public URL could 302 straight to a private address.
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                problem = await _destination_error(target)
                if problem:
                    return problem
                request = client.build_request("GET", target, headers=HEADERS)
                resp = await client.send(request, stream=True)
                if resp.has_redirect_location and resp.next_request is not None:
                    target = resp.next_request.url
                    await resp.aclose()
                    resp = None
                    continue
                break
            else:
                return f"[error] too many redirects (more than {MAX_REDIRECTS}) fetching {url}"

            if resp is None:
                return f"[error] could not fetch {url}"

            try:
                if resp.status_code >= 400:
                    return f"[error] {resp.status_code} {resp.reason_phrase or ''}".rstrip() + f" fetching {resp.url}"
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                source = str(resp.url)
                raw = await _read_capped(resp)
                encoding = resp.encoding or "utf-8"
            finally:
                await resp.aclose()
    except httpx.TimeoutException:
        return f"[error] timed out after 20s fetching {target}"
    except httpx.HTTPError as e:
        return f"[error] could not fetch {target}: {e}"

    body = raw.decode(encoding, "replace")

    title = ""
    if "html" in ctype:
        title = _page_title(body)
        text = html_to_text(body)
    elif not ctype or ctype.startswith("text/") or "json" in ctype or ctype.endswith("+xml"):
        text = body  # already readable -- don't mangle JSON or source files
    else:
        # Binary bodies (images, PDFs, downloads) would just be noise.
        return f"[error] unsupported content type: {ctype}"

    header = source + (f" -- {title}" if title else "")
    return _truncate(f"{header}\n\n{text.strip()}", max_chars)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _redact(text: str, api_key: str) -> str:
    """A key can surface in an httpx error or a provider's echoed request;
    it must never reach a chat bubble."""
    return text.replace(api_key, "***") if api_key else text


def _snippet(value) -> str:
    if not isinstance(value, str):
        return ""
    # Brave wraps query terms in <strong> and escapes entities in both
    # titles and descriptions; neither is useful to a model.
    text = " ".join(html.unescape(TAG_RE.sub("", value)).split())
    if len(text) > SNIPPET_LIMIT:
        text = text[:SNIPPET_LIMIT - 3].rstrip() + "..."
    return text


def _format_results(results: list, title_key: str, snippet_key: str) -> str:
    lines = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = item.get("url") if isinstance(item.get("url"), str) else ""
        title = _snippet(item.get(title_key))
        if not title and not url:
            continue  # nothing to show; numbering counts what's printed, not what was skipped
        entry = f"{len(lines) + 1}. {title or '(untitled)'}\n   {url}"
        snippet = _snippet(item.get(snippet_key))
        if snippet:
            entry += f"\n   {snippet}"
        lines.append(entry)
    return "\n".join(lines) if lines else "(no results)"


async def web_search(query: str, provider: str, api_key: str, limit: int = 5) -> str:
    """Run one search against the provider configured in the UI."""
    name = (provider or "").strip().lower()
    if name not in SEARCH_PROVIDERS:
        return f"[error] unknown search provider '{provider or ''}'"
    api_key = (api_key or "").strip()
    if not api_key:
        return "[error] no search API key configured -- add one in the UI's Agent tab, under Integrations"
    query = (query or "").strip()
    if not query:
        return "[error] no search query given"

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 10))

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if name == "brave":
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": limit},
                    headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                )
            else:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "max_results": limit},
                )
    except httpx.TimeoutException:
        return f"[error] {name} search timed out after 20s"
    except httpx.HTTPError as e:
        return f"[error] could not reach {name}: {_redact(str(e), api_key)}"

    if resp.status_code in (401, 403):
        return f"[error] {name} rejected the API key ({resp.status_code}) -- check it in the UI's Agent tab"
    if resp.status_code == 429:
        return f"[error] {name} rate limit hit (429) -- wait a bit and retry"
    if resp.status_code >= 400:
        return f"[error] {name} returned {resp.status_code}: {_redact(resp.text[:200], api_key)}"

    try:
        data = resp.json()
    except ValueError:
        return f"[error] {name} returned a non-JSON response ({resp.status_code})"
    if not isinstance(data, dict):
        return f"[error] unexpected response from {name}: not a JSON object"

    if name == "brave":
        results = (data.get("web") or {}).get("results") if isinstance(data.get("web"), dict) else None
        title_key, snippet_key = "title", "description"
    else:
        results = data.get("results")
        title_key, snippet_key = "title", "content"
    if results is None:
        return f"[error] unexpected response from {name}: no results field"
    if not isinstance(results, list):
        return f"[error] unexpected response from {name}: results is not a list"

    body = _format_results(results[:limit], title_key, snippet_key)
    return _truncate(_redact(body, api_key), OUTPUT_LIMIT)
