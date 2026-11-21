/*
 * ToRc - torc-server, the chat server
 *
 *                                                                                    ,---,_          ,
 *     Author: Rendeverance                                                           _>   `'-.  .--'/
 *                                                                               .--'` ._      `/   <_
 *                                                                                 >,-' ._'.. ..__ . ' '-.
 *                                                                             .-'   .'`         `'.     '.
 *                                                                              >   / >`-.     .-'< \ , '._\
 *                                                                             /    ; '-._>   <_.-' ;  '._>
 *                                                                            `>  ,/  /___\ /___\  \_  /
 *                                                                             `.-|(|  \o_/  \o_/   |)|`
 *                                                                                 \;        \      ;/
 *                                                                                   \  .-,   )-.  /
 *                                                                                    /`  .'-'.  `\
 *                                                                                   ;_.-`.___.'-.;
 *
 *      torc-server.rs
 *      Big models, small machines.
 */
//! torc-server <model> [port] [threads] [ctx] [name]: a chat server for every model ToRc runs.
//!
//! One conversation at a time, over HTTP: GET /health, /slots, /metrics and /v1/models (also /v1/models/<name>), and
//! POST /v1/chat/completions (OpenAI), /v1/messages (Anthropic) and /v1/responses (OpenAI Responses), each as one JSON
//! reply or as that API's stream of events. While a reply is being written, the requests that don't need the model
//! (/health, /slots, /metrics, the model list) are answered at once, and the others wait their turn in the order
//! their requests arrive in full. GET /torc/settings gives the settings it runs with as a profile, and POST
//! /torc/settings reads one in for its next start (runtime/profile.rs, AUTOTUNE.md). This file is the HTTP side. The
//! model answers through chat/serve.rs, each family from its own folder under models/, picked by the file's
//! architecture.

use model::api;
use model::chat::serve::{Reply, Serve, tool_calls};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

/// OpenAI's usage block. cached_tokens is how many prompt tokens were not read again (from the state kept from the last
/// request, its mark or the system prompt's).
fn usage(np: usize, nr: usize, nt: usize, kept: usize) -> Value {
    json!({"prompt_tokens": np, "completion_tokens": nr, "total_tokens": np + nr,
        "prompt_tokens_details": {"cached_tokens": kept}, "completion_tokens_details": {"reasoning_tokens": nt}})
}

/// The engine's own timing of a reply, in llama-server's form: the prompt tokens read and the time reading them, the
/// reply tokens and the time from the first token going out to the end, and the time to the first token (ttft_ms).
/// No HTTP or client time is in it.
fn timings((pn, ps, ft, rn, rs): (usize, f64, f64, usize, f64)) -> Value {
    let per = |n: usize, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };
    json!({"prompt_n": pn, "prompt_ms": ps * 1e3, "prompt_per_second": per(pn, ps), "ttft_ms": ft * 1e3,
        "predicted_n": rn, "predicted_ms": rs * 1e3, "predicted_per_second": per(rn, rs)})
}

fn respond(s: &mut TcpStream, status: &str, ctype: &str, body: &str) {
    // the charset named: a client that guesses (Windows PowerShell 5.1 reads ISO-8859-1) shows UTF-8 text wrongly
    let _ = write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

/// The largest request body read (a chat with long attachments runs to a few MB).
const MAX_BODY: usize = 16 << 20;

/// A request's first line and headers together may take this many bytes.
const MAX_HEAD: usize = 64 << 10;

/// A whole request must arrive within this long.
const REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);

/// The connection while a request is read. Each read waits at most until the request's deadline (and never more than
/// 10 s), so a client that sends its head a byte at a time can't hold the server past REQUEST_DEADLINE: read_line
/// would otherwise keep waiting as long as each byte comes inside the socket's timeout.
struct Deadline {
    s: TcpStream,
    end: std::time::Instant,
}

impl Read for Deadline {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.end.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return Err(std::io::ErrorKind::TimedOut.into());
        }
        self.s.set_read_timeout(Some(left.min(std::time::Duration::from_secs(10))))?;
        self.s.read(buf)
    }
}

/// One line of a request's head, within what is left of MAX_HEAD. None when the head is too long, the line is cut
/// short or the read fails.
fn head_line(r: &mut BufReader<Deadline>, left: &mut usize) -> Option<String> {
    // nothing left: a head that has used all of MAX_HEAD without ending is too long, not finished
    if *left == 0 {
        return None;
    }
    let mut h = String::new();
    let n = r.by_ref().take(*left as u64).read_line(&mut h).ok()?;
    *left -= n;
    if n > 0 && !h.ends_with('\n') {
        return None;
    }
    Some(h)
}

/// Whether an Origin header's value names this machine (a page served from it), as against another site's page.
fn local_origin(v: &str) -> bool {
    let host = v.trim().split("://").nth(1).unwrap_or("");
    let host = host.split('/').next().unwrap_or("");
    let name =
        if host.starts_with('[') { host.split(']').next().unwrap_or("") } else { host.split(':').next().unwrap_or("") };
    matches!(name, "localhost" | "127.0.0.1" | "[::1")
}

/// The key another machine's requests must carry (TORC_API_KEY), read once at start. None: no key is asked for.
static KEY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// The "created" time /v1/models gives the model, in seconds since 1970: when its file was last written.
static CREATED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// With a key set, how long another machine has to send its request's head: every client sends it at once, and a
/// machine without the key must not hold one of the MAX_OPEN connections for REQUEST_DEADLINE.
const KEYED_HEAD_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// A key one header line carries: "Authorization: Bearer <key>" (OpenAI's clients) or "x-api-key: <key>"
/// (Anthropic's). The header names and "Bearer" in any case; the key as written.
fn carried_key(line: &str) -> Option<&str> {
    let (name, value) = line.split_once(':')?;
    let value = value.trim();
    match name.trim().to_ascii_lowercase().as_str() {
        "x-api-key" => Some(value),
        "authorization" => {
            let (scheme, key) = value.split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then(|| key.trim())
        }
        _ => None,
    }
    .filter(|k| !k.is_empty())
}

/// Whether two keys are the same, in a time that doesn't depend on where they first differ.
fn same_key(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// Whether a request may be served: always without a key set, and always from this machine; from another machine only
/// when one of the keys it carries is the key.
fn allowed(key: Option<&str>, local: bool, given: &[String]) -> bool {
    match key {
        None => true,
        Some(_) if local => true,
        Some(k) => given.iter().any(|g| same_key(g.as_bytes(), k.as_bytes())),
    }
}

/// A connection from this machine: 127.0.0.1 and the rest of 127/8, ::1, and 127.0.0.1 written as IPv6.
fn local_peer(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_loopback(),
        std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback()),
    }
}

/// The refusal of a request without the key, in the error shape of the API it came by, so a client shows it.
fn key_refused(path: &str) -> String {
    let text = "this server needs its API key from other machines: send it as \"Authorization: Bearer <key>\" or \
                \"x-api-key: <key>\"";
    if route(path).starts_with("/v1/messages") {
        json!({"type": "error", "error": {"type": "authentication_error", "message": text}}).to_string()
    } else {
        json!({"error": {"message": text, "type": "invalid_request_error", "code": "invalid_api_key"}}).to_string()
    }
}

/// A request's path as the routes name it: without its query (some clients ask for /v1/messages?beta=true) or a
/// trailing slash, and with the /v1 that clients given a base address without it leave out.
fn route(path: &str) -> String {
    let p = path.split('?').next().unwrap_or("").trim_end_matches('/');
    if p.is_empty() || p == "/v1" || p.starts_with("/v1/") || p.starts_with("/torc/") {
        p.to_string()
    } else {
        format!("/v1{p}")
    }
}

/// The loaded model as /v1/models lists it.
fn model_entry(name: &str) -> Value {
    json!({"id": name, "object": "model", "created": CREATED.get().copied().unwrap_or(0), "owned_by": "torc"})
}

/// A path segment with its %XX escapes undone ("Llama%203" is "Llama 3"); a bad escape is kept as written.
fn unescape(segment: &str) -> String {
    let b = segment.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2]))
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| segment.to_string())
}

/// Reads and drops a served request's body that the route doesn't use (GET /health with one).
fn drain(r: &mut BufReader<Deadline>, len: usize) {
    let _ = std::io::copy(&mut r.by_ref().take(len.min(MAX_BODY) as u64), &mut std::io::sink());
}

/// How long a refused request's client is read after the refusal is sent.
const LINGER: std::time::Duration = std::time::Duration::from_secs(5);

/// Refuses a request so that the client gets the reason: the reply goes first, then the sending side is closed, and
/// what the client still sends (a body of any length, by Content-Length or chunked) is read and dropped until it
/// closes, for LINGER at most. Closing a socket with data still unread resets the connection on Windows, and a reset
/// can reach the client before the reply does, which then reports "connection reset" instead of the reason.
fn refuse(s: &mut TcpStream, r: &mut BufReader<Deadline>, status: &str, body: &str) {
    respond(s, status, "application/json", body);
    let _ = s.flush();
    let _ = s.shutdown(std::net::Shutdown::Write);
    r.get_mut().end = std::time::Instant::now() + LINGER;
    let _ = std::io::copy(&mut r.by_ref().take(MAX_BODY as u64), &mut std::io::sink());
}

/// The most connections open at once, those waiting for the model included (each holds what it has sent of its
/// request until the model gets to it). One more is told the server is busy. 128 is the queue the operating system
/// kept for the server before 0.1.6, when every connection waited there for its turn.
const MAX_OPEN: usize = 128;

/// The most busy refusals being sent at once; past that a connection over MAX_OPEN is closed unanswered.
const MAX_REFUSING: usize = 32;

/// The connections open now, and the busy refusals being sent.
static OPEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static REFUSING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// One of the refusals REFUSING counts, given back when it is dropped (also when its thread could not be started).
struct Refusing;

impl Drop for Refusing {
    fn drop(&mut self) {
        REFUSING.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// One of the connections OPEN counts, given back when it is dropped: when its request has been answered, whichever
/// thread answered it.
struct Open;

impl Drop for Open {
    fn drop(&mut self) {
        OPEN.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The most request body bytes held at once, by the requests being read and those waiting for the model: 16 bodies of
/// the largest size. Past it a request is told the server is busy.
const MAX_HELD: usize = 16 * MAX_BODY;

/// The request body bytes held now (MAX_HELD).
static HELD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A request's share of HELD: the body bytes it has received, given back when it is dropped (when the request has been
/// answered).
struct Held(usize);

impl Held {
    /// Adds n received bytes to the share, or says no (and adds nothing) when that would take HELD past MAX_HELD.
    fn grow(&mut self, n: usize) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        if HELD.fetch_add(n, SeqCst) + n > MAX_HELD {
            HELD.fetch_sub(n, SeqCst);
            return false;
        }
        self.0 += n;
        true
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(self.0, std::sync::atomic::Ordering::SeqCst);
    }
}

/// A request that needs the model, read and let in, waiting its turn on the model's thread.
struct Job {
    method: String,
    path: String,
    body: Vec<u8>,
    s: TcpStream,
    // whether another request was being answered or waiting when this one came: only then can its client have
    // given up waiting
    waited: bool,
    _open: Open,
    _held: Held,
}

/// The loaded model's name and context, for the routes answered off the model's thread.
static SERVED: std::sync::OnceLock<(String, usize)> = std::sync::OnceLock::new();

/// What the server is doing and has done, for /slots and /metrics: the model's thread writes it, the connections'
/// threads read it.
#[derive(Default)]
struct Status {
    /// a reply is being written (a token count or a settings request doesn't count)
    busy: bool,
    /// the number of the reply being written, or of the last one
    task: u64,
    /// requests let in and waiting for the model
    waiting: usize,
    /// the current reply's sampling settings, in llama-server's names
    params: Value,
    /// the current reply's pieces of text sent so far (about one a token)
    decoded: u64,
    /// since the server started: the prompt tokens read, those kept from before, and the seconds reading; the reply
    /// tokens and their seconds; the longest prompt and reply together
    prompt: u64,
    cached: u64,
    prompt_s: f64,
    written: u64,
    written_s: f64,
    longest: u64,
    /// the prompt tokens read and their seconds, and the reply tokens and theirs, since /metrics was last read
    window: (u64, f64, u64, f64),
}

static STATUS: std::sync::LazyLock<std::sync::Mutex<Status>> = std::sync::LazyLock::new(Default::default);

fn state() -> std::sync::MutexGuard<'static, Status> {
    STATUS.lock().unwrap_or_else(|p| p.into_inner())
}

/// GET /slots, in llama-server's shape: one slot, since one reply is written at a time.
fn slots() -> Value {
    let st = state();
    let ctx = SERVED.get().map_or(0, |s| s.1);
    let max = st.params["n_predict"].as_i64().unwrap_or(-1);
    let remain = if st.busy && max >= 0 { (max - st.decoded as i64).max(0) } else { -1 };
    json!([{"id": 0, "id_task": st.task, "n_ctx": ctx, "is_processing": st.busy, "params": st.params,
        "next_token": {"has_next_token": st.busy, "n_remain": remain, "n_decoded": st.decoded}}])
}

/// GET /metrics: Prometheus' text, under llama-server's names, so that a monitor made for it reads this too. Its
/// tokens a second are over the time since the last read, from the replies that finished in it.
fn metrics() -> String {
    let mut st = state();
    let (pn, ps, wn, ws) = std::mem::take(&mut st.window);
    let per = |n: u64, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };
    let counters = [
        ("prompt_tokens_total", "Number of prompt tokens processed, excluding cached tokens", st.prompt as f64),
        ("prompt_tokens_cached_total", "Number of prompt tokens reused from the cache", st.cached as f64),
        ("prompt_seconds_total", "Total time spent processing prompts", st.prompt_s),
        ("tokens_predicted_total", "Number of generation tokens processed", st.written as f64),
        ("tokens_predicted_seconds_total", "Total time spent generating tokens", st.written_s),
        ("n_tokens_max", "Largest observed sequence length (prompt + generation)", st.longest as f64),
    ];
    let gauges = [
        ("prompt_tokens_seconds", "Average prompt throughput in tokens/s", per(pn, ps)),
        ("predicted_tokens_seconds", "Average generation throughput in tokens/s", per(wn, ws)),
        ("requests_processing", "Number of requests processing", if st.busy { 1.0 } else { 0.0 }),
        ("requests_deferred", "Number of requests deferred", st.waiting as f64),
    ];
    let mut out = String::new();
    for (kind, items) in [("counter", &counters[..]), ("gauge", &gauges[..])] {
        for (name, help, v) in items {
            out += &format!("# HELP llamacpp:{name} {help}\n# TYPE llamacpp:{name} {kind}\nllamacpp:{name} {v}\n");
        }
    }
    out
}

/// Whether a request's query has a parameter with a value (llama-server's test for ?fail_on_no_slot).
fn has_param(path: &str, name: &str) -> bool {
    let query = path.split_once('?').map_or("", |q| q.1);
    query.split('&').any(|p| p.split_once('=').is_some_and(|(k, v)| k == name && !v.is_empty()))
}

/// The model, with what it does noted in STATUS as it goes.
struct Noted<'a>(&'a mut dyn Serve);

impl Serve for Noted<'_> {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn count(&self, c: &api::Chat) -> usize {
        self.0.count(c)
    }

    fn reply(&mut self, c: &api::Chat, emit: &mut dyn FnMut(bool, &str) -> bool) -> format::Result<Reply> {
        {
            let mut st = state();
            st.busy = true;
            st.task += 1;
            st.decoded = 0;
            st.params = json!({"n_predict": c.max, "max_tokens": c.max, "seed": c.seed, "temperature": c.temperature,
                "top_k": c.top_k.unwrap_or(40), "top_p": c.top_p, "presence_penalty": c.presence_penalty,
                "frequency_penalty": c.frequency_penalty, "stream": c.stream});
        }
        let r = self.0.reply(c, &mut |reasoning, t| {
            if !t.is_empty() {
                state().decoded += 1;
            }
            emit(reasoning, t)
        });
        let (pn, ps, _, wn, ws) = self.0.timed();
        let kept = self.0.kept();
        let mut st = state();
        st.busy = false;
        let r = r?;
        st.prompt += pn as u64;
        st.cached += kept as u64;
        st.prompt_s += ps;
        st.written += wn as u64;
        st.written_s += ws;
        st.longest = st.longest.max((r.prompt + r.output) as u64);
        st.window.0 += pn as u64;
        st.window.1 += ps;
        st.window.2 += wn as u64;
        st.window.3 += ws;
        Ok(r)
    }

    fn kept(&self) -> usize {
        self.0.kept()
    }

    fn timed(&self) -> (usize, f64, f64, usize, f64) {
        self.0.timed()
    }

    fn replies(&self) -> u64 {
        self.0.replies()
    }

    fn idle(&mut self) {
        self.0.idle()
    }
}

/// A connection, on a thread of its own: reads the request and answers it there when it doesn't need the model
/// (/health, /slots, /metrics, the model list), so those are answered even while a reply is being written. A request
/// that needs the model goes to the model's thread (jobs) and waits its turn, in the order requests arrive in full.
fn handle(mut s: TcpStream, jobs: &std::sync::mpsc::Sender<Job>, open: Open) {
    // a client that stalls before finishing its request must not hold its thread for long (the whole request has
    // REQUEST_DEADLINE, see Deadline), and nor must one that stops reading the reply without closing (a write that
    // can't go out in a minute ends it)
    let _ = s.set_write_timeout(Some(std::time::Duration::from_secs(60)));
    let Ok(rs) = s.try_clone() else { return };
    // with TORC_API_KEY set, another machine's request needs the key, and gets only a few seconds to send its head;
    // this machine's never needs it (a failed address lookup counts as another machine)
    let key = KEY.get().and_then(|k| k.as_deref());
    let local = s.peer_addr().map(|a| local_peer(a.ip())).unwrap_or(false);
    let start = std::time::Instant::now();
    let head_end = if key.is_some() && !local { KEYED_HEAD_DEADLINE } else { REQUEST_DEADLINE };
    let mut r = BufReader::new(Deadline { s: rs, end: start + head_end });
    let mut left = MAX_HEAD;
    let Some(line) = head_line(&mut r, &mut left) else { return };
    let (method, raw) = {
        let mut it = line.split_whitespace();
        (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
    };
    let (mut len, mut foreign) = (0usize, false);
    let mut given = Vec::new();
    loop {
        let Some(h) = head_line(&mut r, &mut left) else { return };
        if h == "\r\n" || h.is_empty() {
            break;
        }
        let low = h.to_ascii_lowercase();
        if let Some(v) = low.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
        if let Some(v) = low.strip_prefix("origin:") {
            foreign = !local_origin(v);
        }
        if let Some(k) = carried_key(&h) {
            given.push(k.to_string());
        }
    }
    // a browser names the page a request comes from, and another site's page is refused, whatever address the server
    // listens on: it could otherwise run the model from any open tab. Clients that aren't web pages send no Origin
    if foreign {
        let why = json!({"error": "requests from another site's web page are refused"}).to_string();
        return refuse(&mut s, &mut r, "403 Forbidden", &why);
    }
    // whether the server is up, for monitors and frameworks that probe before they start: no key needed (it says
    // nothing but that), as llama-server's /health
    if method == "GET" && route(&raw) == "/v1/health" {
        drain(&mut r, len);
        respond(&mut s, "200 OK", "application/json", r#"{"status":"ok"}"#);
        return;
    }
    if !allowed(key, local, &given) {
        return refuse(&mut s, &mut r, "401 Unauthorized", &key_refused(&raw));
    }
    // let in: the rest of the request has what is left of the whole request's time
    r.get_mut().end = start + REQUEST_DEADLINE;
    if len > MAX_BODY {
        let why = json!({"error": format!("request body over {} MB", MAX_BODY >> 20)}).to_string();
        return refuse(&mut s, &mut r, "413 Payload Too Large", &why);
    }
    // read as it arrives: memory is taken for what a client has sent, not for the length it says it will send. The
    // bodies held at once, being read or waiting their turn, are capped (MAX_HELD) and counted as they arrive, so a
    // burst of large requests can't take gigabytes on top of the model, and a client that only says it will send a
    // large body holds none of the cap
    let mut held = Held(0);
    let mut body = Vec::new();
    let mut chunk = vec![0u8; 64 << 10];
    while body.len() < len {
        let want = (len - body.len()).min(chunk.len());
        let n = match r.read(&mut chunk[..want]) {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        if !held.grow(n) {
            let why = json!({"error": {"code": 503, "message": "the server is busy: too many requests waiting, try again shortly",
                "type": "unavailable_error"}});
            return refuse(&mut s, &mut r, "503 Service Unavailable", &why.to_string());
        }
        body.extend_from_slice(&chunk[..n]);
    }
    let path = route(&raw);
    let name = SERVED.get().map_or("", |m| m.0.as_str());
    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/models") => {
            let j = json!({"object": "list", "data": [model_entry(name)]});
            respond(&mut s, "200 OK", "application/json", &j.to_string());
        }
        // one model by its name, as OpenAI's clients ask before they use it
        ("GET", p) if p.starts_with("/v1/models/") => {
            let asked = unescape(&p["/v1/models/".len()..]);
            if asked == name {
                respond(&mut s, "200 OK", "application/json", &model_entry(name).to_string());
            } else {
                let text = format!("no model {asked} here: this server runs {name}");
                let j = json!({"error": {"message": text, "type": "invalid_request_error", "code": "model_not_found"}});
                respond(&mut s, "404 Not Found", "application/json", &j.to_string());
            }
        }
        // what the server is doing, for monitors: as llama-server answers, ?fail_on_no_slot=1 makes a busy server a 503
        ("GET", "/v1/slots") => {
            if has_param(&raw, "fail_on_no_slot") && state().busy {
                let j = json!({"error": {"code": 503, "message": "no slot available", "type": "unavailable_error"}});
                respond(&mut s, "503 Service Unavailable", "application/json", &j.to_string());
            } else {
                respond(&mut s, "200 OK", "application/json", &slots().to_string());
            }
        }
        ("GET", "/v1/metrics") => respond(&mut s, "200 OK", "text/plain; version=0.0.4", &metrics()),
        ("GET", "/torc/settings")
        | ("POST", "/torc/settings" | "/v1/chat/completions" | "/v1/messages" | "/v1/responses")
        | ("POST", "/v1/messages/count_tokens") => {
            let waited = {
                let mut st = state();
                st.waiting += 1;
                st.busy || st.waiting > 1
            };
            if jobs.send(Job { method, path, body, s, waited, _open: open, _held: held }).is_err() {
                state().waiting -= 1;
            }
        }
        _ => respond(&mut s, "404 Not Found", "application/json", r#"{"error":"not found"}"#),
    }
}

/// Whether a waiting request's client has gone: it closed the connection, or it was reset, while the request waited
/// for its turn. A client still there has sent nothing more, so a look at what it sent finds nothing to read yet.
fn gone(s: &TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() {
        return false;
    }
    let r = s.peek(&mut [0u8; 1]);
    let _ = s.set_nonblocking(false);
    match r {
        Ok(n) => n == 0,
        Err(err) => matches!(
            err.kind(),
            std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::BrokenPipe
        ),
    }
}

/// A request on the model's thread: the routes that need the model, or that change what it starts with next.
fn run(e: &mut dyn Serve, j: Job) {
    let Job { method, path, body, mut s, waited, .. } = j;
    let s = &mut s;
    match (method.as_str(), path.as_str()) {
        ("GET", "/torc/settings") => {
            respond(s, "200 OK", "application/json", &model::profile::export(e.name()).to_string());
        }
        ("POST", "/torc/settings") => {
            let Ok(p) = serde_json::from_slice::<Value>(&body) else {
                return respond(s, "400 Bad Request", "application/json", r#"{"error":"bad json"}"#);
            };
            match model::profile::import(&p) {
                Ok((layout, done)) => {
                    let j = json!({"ok": true, "written": done, "layout": layout, "running": model::tune::layout(),
                        "note": "used from the server's next start in that layout"});
                    respond(s, "200 OK", "application/json", &j.to_string());
                }
                Err(err) => {
                    respond(s, "400 Bad Request", "application/json", &json!({"error": err}).to_string());
                }
            }
        }
        _ => {
            // a client that gave up while it waited would have its reply written for nobody (only a request that waited
            // is looked at: a client that sends its request and then closes its side, as some scripts do, is still
            // there to read the reply, and to a look it seems gone)
            if waited && gone(s) {
                eprintln!("torc-server: a waiting request's client left before its turn: not started");
                return;
            }
            let Ok(req) = serde_json::from_slice::<Value>(&body) else {
                return respond(s, "400 Bad Request", "application/json", r#"{"error":"bad json"}"#);
            };
            let (a, c) = match path.as_str() {
                "/v1/chat/completions" => (Api::Chat, api::from_chat_completions(&req)),
                "/v1/responses" => (Api::Responses, api::from_responses(&req)),
                _ => (Api::Messages, api::from_messages(&req)),
            };
            if path == "/v1/messages/count_tokens" {
                let n = e.count(&c);
                return respond(s, "200 OK", "application/json", &json!({"input_tokens": n}).to_string());
            }
            complete(e, s, a, c);
        }
    }
}

/// The API a request came in by, which decides the shape of its reply.
#[derive(Clone, Copy, PartialEq)]
enum Api {
    /// OpenAI chat completions
    Chat,
    /// Anthropic Messages
    Messages,
    /// OpenAI Responses
    Responses,
}

/// The HTTP status for a reply that failed: a prompt that does not fit the context is the request's fault, anything
/// else the server's.
fn status(e: &format::Error) -> &'static str {
    if model::chat::serve::is_too_long(e) { "400 Bad Request" } else { "500 Internal Server Error" }
}

/// Runs a request and writes its reply in the shape of the API it came in by: one JSON body, or that API's stream of
/// events.
fn complete(e: &mut dyn Serve, s: &mut TcpStream, a: Api, c: api::Chat) {
    let name = e.name().to_string();
    if c.stream {
        let _ = write!(
            s,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        );
    }
    let Ok(mut out) = s.try_clone() else { return };
    // a failed write means the client closed the stream, and run then stops the reply
    let mut send = |t: &str| t.is_empty() || write!(out, "{t}").is_ok() && out.flush().is_ok();
    match a {
        Api::Chat => {
            let id = format!(
                "chatcmpl-{}",
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis())
            );
            let chunk = |delta: Value, fin: Value| {
                json!({"id": id, "object": "chat.completion.chunk", "model": name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
            };
            if c.stream {
                send(&format!("data: {}\n\n", chunk(json!({"role": "assistant"}), Value::Null)));
                let r = e.reply(&c, &mut |rs, t| {
                    let d = if rs { json!({"reasoning_content": t}) } else { json!({"content": t}) };
                    send(&format!("data: {}\n\n", chunk(d, Value::Null)))
                });
                match r {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        if !calls.is_empty() {
                            send(&format!(
                                "data: {}\n\n",
                                chunk(json!({"tool_calls": tool_calls(&calls)}), Value::Null)
                            ));
                        }
                        send(&format!("data: {}\n\n", chunk(json!({}), json!(fin))));
                        // the usage chunk (stream_options.include_usage) has no choices, as OpenAI sends it
                        let u = json!({"id": id, "object": "chat.completion.chunk", "model": name, "choices": [],
                            "usage": usage(np, nr, nt, e.kept()), "timings": timings(e.timed())});
                        send(&format!("data: {u}\n\n"));
                    }
                    Err(err) => {
                        send(&format!(
                            "data: {}\n\n",
                            json!({"error": {"message": err.to_string(), "type": "server_error"}})
                        ));
                    }
                }
                send("data: [DONE]\n\n");
            } else {
                let (mut reply, mut reasoning) = (String::new(), String::new());
                match e.reply(&c, &mut |rs, t| {
                    if rs {
                        reasoning.push_str(t)
                    } else {
                        reply.push_str(t)
                    }
                    true
                }) {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        let mut msg = json!({"role": "assistant", "content": reply});
                        if !reasoning.is_empty() {
                            msg["reasoning_content"] = json!(reasoning);
                        }
                        if !calls.is_empty() {
                            msg["tool_calls"] = tool_calls(&calls);
                        }
                        let j = json!({"id": id, "object": "chat.completion", "model": name,
                            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                            "usage": usage(np, nr, nt, e.kept()), "timings": timings(e.timed())});
                        respond(s, "200 OK", "application/json", &j.to_string());
                    }
                    Err(err) => {
                        respond(s, status(&err), "application/json", &json!({"error": err.to_string()}).to_string())
                    }
                }
            }
        }
        Api::Messages => {
            let u = |np, nr, nt, kept| api::Usage { prompt: np, cached: kept, output: nr, reasoning: nt };
            if c.stream {
                let mut ms = api::MessagesStream::new(c.show_reasoning);
                let np0 = e.count(&c);
                send(&ms.start(&name, np0));
                let r = e.reply(&c, &mut |rs, t| send(&ms.delta(rs, t)));
                match r {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        send(&ms.finish(&tool_calls(&calls), fin, &u(np, nr, nt, e.kept())));
                    }
                    Err(err) => {
                        send(&api::MessagesStream::error(&err.to_string()));
                    }
                }
            } else {
                let (mut reply, mut reasoning) = (String::new(), String::new());
                match e.reply(&c, &mut |rs, t| {
                    if rs {
                        reasoning.push_str(t)
                    } else {
                        reply.push_str(t)
                    }
                    true
                }) {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        let j = api::messages_reply(
                            &name,
                            &reasoning,
                            &reply,
                            &tool_calls(&calls),
                            fin,
                            &u(np, nr, nt, e.kept()),
                            c.show_reasoning,
                        );
                        respond(s, "200 OK", "application/json", &j.to_string());
                    }
                    Err(err) => respond(
                        s,
                        status(&err),
                        "application/json",
                        &api::messages_error("api_error", &err.to_string()).to_string(),
                    ),
                }
            }
        }
        Api::Responses => {
            let mut rs = api::ResponsesStream::new(&name, c.stream, c.show_reasoning);
            send(&rs.start());
            let r = e.reply(&c, &mut |reasoning, t| send(&rs.delta(reasoning, t)));
            match r {
                Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                    let u = api::Usage { prompt: np, cached: e.kept(), output: nr, reasoning: nt };
                    let body = rs.finish(&tool_calls(&calls), fin, &u);
                    if c.stream {
                        send(&body);
                    } else {
                        respond(s, "200 OK", "application/json", &body);
                    }
                }
                Err(err) if c.stream => {
                    send(&rs.error(&err.to_string()));
                }
                Err(err) => respond(
                    s,
                    status(&err),
                    "application/json",
                    &json!({"error": {"message": err.to_string(), "type": "server_error"}}).to_string(),
                ),
            }
        }
    }
}

fn main() {
    model::set_npu_env();
    // every panic's message and backtrace is appended to panic.log next to the executable, and a panic on the main
    // thread (where the model answers requests) ends the server there and then, rather than leaving the process alive
    // with nothing in its log. Another thread's panic carries on as before: a connection's thread closes only that
    // connection, and the NPU helper's ends only its request
    let dflt = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        dflt(info);
        let bt = std::backtrace::Backtrace::force_capture();
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let name = std::thread::current().name().unwrap_or("?").to_string();
        if let Some(p) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("panic.log")))
            && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p)
        {
            let _ = writeln!(f, "{t} {name}: {info}\n{bt}");
        }
        if name == "main" {
            std::process::exit(101);
        }
    }));
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: torc-server <model.torc> [port] [threads|auto] [ctx] [name]");
        eprintln!(
            "       torc-server <model.torc> --tokenize <text>   (the prompt's ids as the server makes them, BOS first)"
        );
        eprintln!("       torc-server --archs                          (the architectures it runs, one line)");
        std::process::exit(2);
    }
    if a[1] == "--archs" {
        println!("{}", model::models::ARCHS.join(" "));
        return;
    }
    // --tokenize: the ids of a text, special tokens matched, with the model's BOS when it has one (for checking the
    // tokenizer against llama-tokenize)
    if a.get(2).map(String::as_str) == Some("--tokenize") {
        let tok = model::tok::Tokenizer::open(&a[1]).unwrap_or_else(|e| {
            eprintln!("torc-server: {e}");
            std::process::exit(1)
        });
        let text = a.get(3).cloned().unwrap_or_default().replace("\\n", "\n");
        match tok.encode(&text, true) {
            Ok(ids) => println!("{:?}", tok.bos().into_iter().chain(ids).collect::<Vec<u32>>()),
            Err(e) => {
                eprintln!("torc-server: {e}");
                std::process::exit(1)
            }
        }
        return;
    }
    let port: u16 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(8080);
    if !cfg!(feature = "gpu") && model::switches::on("TORC_GPU_ATTN") {
        eprintln!("torc-server: TORC_GPU_ATTN is set, but this build has no GPU tier (--features torc-model/gpu)");
    }
    // threads: a number, or auto (or nothing) for every CPU of the fastest core class (pool.rs), never more than there
    // are CPUs
    let asked: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let threads = model::pool::init(asked).threads as i32;
    eprintln!("torc-server: threads {threads}{}", if asked == 0 { " (auto: the fastest cores)" } else { "" });
    let ctx: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(8192);
    // each family answers through its own module (serve.rs), chosen by the file's architecture
    let meta = format::read::Model::open(&a[1]).ok().and_then(|f| format::meta::Meta::parse(&f.metadata).ok());
    let arch = meta.as_ref().and_then(|mm| mm.str("general.architecture").map(str::to_string));
    // a context past the one the model was made for runs, but its replies get worse past that point (llama.cpp refuses
    // it): say so, once, where the log shows it
    if let Some(trained) = meta.as_ref().and_then(|mm| mm.arch_u32("context_length"))
        && ctx > trained as usize
    {
        eprintln!(
            "torc-server: context {ctx} is past the {trained} tokens this model was made for: past that, replies get worse"
        );
    }
    let (path, name) = (a[1].as_str(), a.get(5).cloned());
    // the model file's own date, so the same model keeps the same "created" from one start to the next
    let written = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let since = written.and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    let _ = CREATED.set(since);
    let mut e = match arch.as_deref() {
        Some("gemma4") => model::models::gemma::serve::open(path, ctx, name),
        Some("gpt-oss") => model::models::gptoss::serve::open(path, ctx, name),
        Some("inkling2_loop") => model::models::inkling2::serve::open(path, ctx, name),
        // every other architecture is one of Model's (models/qwen), which refuses a file it can't run
        _ => model::models::qwen::serve::open(path, ctx, name),
    };
    // the pinning torc-tune chose for this model on this machine, now that its tune file can be found
    model::pool::apply_tuned_pin();
    listen(&mut *e, port, ctx);
}

/// Serves requests on the port until the process ends. Connections are taken on a thread of their own, each read on
/// its own thread (handle), and the model works on one request at a time here, on the main thread, in the order their
/// requests arrive in full.
fn listen(e: &mut dyn Serve, port: u16, ctx: usize) {
    // this machine only, unless TORC_HOST says otherwise; then TORC_API_KEY, when set, is what another machine's
    // requests must carry (this machine's never need it)
    let host = model::switches::get("TORC_HOST").unwrap_or_else(|| "127.0.0.1".into());
    let key = model::switches::get("TORC_API_KEY").map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    let keyed = key.is_some();
    let _ = KEY.set(key);
    let _ = SERVED.set((e.name().to_string(), ctx));
    let l = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|err| {
        eprintln!("torc-server: cannot listen on {host}:{port}: {err}");
        std::process::exit(1)
    });
    let needs = if keyed { ", other machines need the API key" } else { "" };
    eprintln!("torc-server: {} on {host}:{port}{needs}", e.name());
    let (jobs, queue) = std::sync::mpsc::channel::<Job>();
    let accepting = std::thread::Builder::new().name("accept".into()).spawn(move || {
        use std::sync::atomic::Ordering::SeqCst;
        for c in l.incoming() {
            let c = match c {
                Ok(c) => c,
                Err(err) => {
                    eprintln!("torc-server: accept: {err}");
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    continue;
                }
            };
            // too many open: this one is told the server is busy, on a thread of its own so that the refusal's wait
            // for the rest of the request holds nothing up; past that many refusals it is closed unanswered
            if OPEN.fetch_add(1, SeqCst) >= MAX_OPEN {
                OPEN.fetch_sub(1, SeqCst);
                if REFUSING.fetch_add(1, SeqCst) < MAX_REFUSING {
                    let refusing = Refusing;
                    let _ = std::thread::Builder::new().name("busy".into()).spawn(move || {
                        let _refusing = refusing;
                        busy(c);
                    });
                } else {
                    REFUSING.fetch_sub(1, SeqCst);
                }
                continue;
            }
            let open = Open;
            let jobs = jobs.clone();
            let _ = std::thread::Builder::new().name("conn".into()).spawn(move || handle(c, &jobs, open));
        }
    });
    if let Err(err) = accepting {
        eprintln!("torc-server: cannot start taking connections: {err}");
        std::process::exit(1)
    }
    // between requests: after TORC_IDLE_SAVE seconds without one (default 60, 0 = never) the current chat goes to disk
    // as well (save_leaving), so a reload or a restart comes back to it warm
    let idle = model::switches::num::<u64>("TORC_IDLE_SAVE").unwrap_or(60);
    let mut e = Noted(e);
    let mut last = std::time::Instant::now();
    loop {
        match queue.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(job) => {
                {
                    let mut st = state();
                    st.waiting = st.waiting.saturating_sub(1);
                }
                // only a reply counts: a client's model polls between chats do not keep the chat off the disk
                let n = e.replies();
                run(&mut e, job);
                if e.replies() != n {
                    last = std::time::Instant::now();
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if idle > 0 && last.elapsed().as_secs() >= idle {
                    e.idle();
                    last = std::time::Instant::now() + std::time::Duration::from_secs(86400 * 365);
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!("torc-server: no longer taking connections");
                std::process::exit(1)
            }
        }
    }
}

/// Tells a connection over MAX_OPEN that the server is busy, so that its client can try again, and reads what it
/// sends for a moment so that the refusal arrives rather than a reset (see refuse).
fn busy(mut s: TcpStream) {
    let Ok(rs) = s.try_clone() else { return };
    let mut r = BufReader::new(Deadline { s: rs, end: std::time::Instant::now() + LINGER });
    let j = json!({"error": {"code": 503, "message": "the server is busy: too many requests waiting, try again shortly",
        "type": "unavailable_error"}});
    refuse(&mut s, &mut r, "503 Service Unavailable", &j.to_string());
}

// The server's guards that need no socket: which pages count as this machine's, and which errors are the caller's.
#[cfg(test)]
mod guards {
    use super::*;

    #[test]
    fn only_this_machine_counts_as_local() {
        for o in [
            "http://localhost",
            "http://localhost:8080",
            "http://127.0.0.1:8080/chat",
            "http://[::1]:8080",
            " http://localhost ",
        ] {
            assert!(local_origin(o), "{o}");
        }
        for o in [
            "https://example.com",
            "http://localhost.example.com",
            "http://127.0.0.1.example.com:8080",
            "null",
            "",
            "file://",
            "http://[::2]:80",
        ] {
            assert!(!local_origin(o), "{o}");
        }
    }

    #[test]
    fn a_prompt_over_the_context_is_the_callers_error() {
        assert_eq!(status(&model::chat::serve::too_long(9000, 8192)), "400 Bad Request");
        assert_eq!(status(&format::Error("the NPU went away".into())), "500 Internal Server Error");
        // an engine fault that happens to mention the context is still the server's
        assert_eq!(status(&format::Error("position past the context".into())), "500 Internal Server Error");
    }

    #[test]
    fn keys_are_read_the_two_standard_ways() {
        assert_eq!(carried_key("Authorization: Bearer abc-123\r\n"), Some("abc-123"));
        assert_eq!(carried_key("authorization: bearer abc-123"), Some("abc-123"));
        assert_eq!(carried_key("x-api-key: Abc-123\r\n"), Some("Abc-123"));
        assert_eq!(carried_key("X-Api-Key:abc"), Some("abc"));
        for h in ["Authorization: Basic abc", "Authorization: Bearer ", "x-api-key: ", "Content-Length: 5", "Bearer x"]
        {
            assert_eq!(carried_key(h), None, "{h}");
        }
    }

    #[test]
    fn only_another_machine_needs_the_key() {
        let k = |s: &str| vec![s.to_string()];
        // no key set: everyone
        assert!(allowed(None, false, &[]));
        // this machine: never asked
        assert!(allowed(Some("s3cret"), true, &[]));
        // another machine: only with the key, exactly
        assert!(!allowed(Some("s3cret"), false, &[]));
        assert!(!allowed(Some("s3cret"), false, &k("s3cre")));
        assert!(!allowed(Some("s3cret"), false, &k("S3CRET")));
        assert!(!allowed(Some("s3cret"), false, &k("s3cret ")));
        assert!(allowed(Some("s3cret"), false, &k("s3cret")));
        assert!(allowed(Some("s3cret"), false, &["wrong".into(), "s3cret".into()]));
    }

    #[test]
    fn this_machine_is_loopback_only() {
        for ip in ["127.0.0.1", "127.8.0.2", "::1", "::ffff:127.0.0.1"] {
            assert!(local_peer(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["192.168.1.20", "10.0.0.1", "100.80.236.60", "::ffff:192.168.1.20", "fe80::1", "0.0.0.0"] {
            assert!(!local_peer(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn paths_fold_the_usual_variations() {
        assert_eq!(route("/v1/models"), "/v1/models");
        assert_eq!(route("/v1/models/"), "/v1/models");
        assert_eq!(route("/models"), "/v1/models");
        assert_eq!(route("/models/qwen?x=1"), "/v1/models/qwen");
        assert_eq!(route("/v1/messages?beta=true"), "/v1/messages");
        assert_eq!(route("/chat/completions"), "/v1/chat/completions");
        assert_eq!(route("/torc/settings"), "/torc/settings");
        assert_eq!(route("/"), "");
        assert_eq!(route("/health"), "/v1/health");
        assert_eq!(route("/v1/health/"), "/v1/health");
        // the refusal comes in the shape of the API asked
        assert!(key_refused("/v1/messages").contains("authentication_error"));
        assert!(key_refused("/v1/chat/completions").contains("invalid_api_key"));
        assert!(key_refused("/messages?beta=true").contains("authentication_error"));
    }

    #[test]
    fn metrics_read_as_llama_servers_do() {
        let text = metrics();
        for name in ["prompt_tokens_total", "tokens_predicted_total", "requests_processing", "requests_deferred"] {
            assert!(text.contains(&format!("\n# TYPE llamacpp:{name} ")), "{name}: {text}");
        }
        // every sample line is a name and a number, after its HELP and TYPE lines
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let (n, v) = l.split_once(' ').unwrap();
            assert!(n.starts_with("llamacpp:") && v.parse::<f64>().is_ok(), "{l}");
        }
        assert_eq!(slots()[0]["id"], 0);
        assert!(slots()[0]["next_token"]["n_decoded"].is_u64());
    }

    #[test]
    fn a_query_parameter_needs_a_value() {
        assert!(has_param("/slots?fail_on_no_slot=1", "fail_on_no_slot"));
        assert!(has_param("/slots?a=b&fail_on_no_slot=true", "fail_on_no_slot"));
        for p in ["/slots", "/slots?fail_on_no_slot", "/slots?fail_on_no_slot=", "/slots?no_fail_on_no_slot=1"] {
            assert!(!has_param(p, "fail_on_no_slot"), "{p}");
        }
    }

    #[test]
    fn model_names_lose_their_escapes() {
        assert_eq!(unescape("Llama%203%208B"), "Llama 3 8B");
        assert_eq!(unescape("qwen3.6-35b-a3b-q4_0"), "qwen3.6-35b-a3b-q4_0");
        assert_eq!(unescape("a%2Fb%2fc"), "a/b/c");
        // a bad or cut-off escape stays as written
        for s in ["100%", "%zz", "a%2", "%"] {
            assert_eq!(unescape(s), s, "{s}");
        }
    }

    /// The bodies held at once stop at MAX_HELD, and a request's share is given back when it is done.
    #[test]
    fn held_bodies_are_capped() {
        let (mut a, mut b) = (Held(0), Held(0));
        assert!(a.grow(MAX_HELD - 10), "under the cap");
        assert!(!b.grow(11), "past the cap");
        assert!(b.grow(10), "up to the cap");
        drop((a, b));
        let mut c = Held(0);
        assert!(c.grow(MAX_HELD), "given back");
    }
}
