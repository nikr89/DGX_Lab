#!/usr/bin/env python3
"""vllm_token_tracker.py - Token-Zähler für alle vLLM-Sessions auf einem Rechner (z. B. DGX Spark).

Prinzip: Jeder vLLM-Server zählt seine Tokens selbst und legt sie unter /metrics offen
(vllm:prompt_tokens_total, vllm:prompt_tokens_cached_total, vllm:generation_tokens_total). Der
Collector findet alle laufenden vLLM-Instanzen automatisch (native Prozesse und Docker-Container),
liest diese Zähler alle paar Sekunden und schreibt die Differenzen in eine SQLite-Datenbank. Eine
"Session" ist ein Server-Lauf: Endpoint + Modell + Startzeitpunkt. Nur Python-Standardbibliothek,
keine Abhängigkeiten.

prompt_tokens_cached ist eine Teilmenge von prompt_tokens: Prompt-Tokens, die aus dem
Prefix-Cache kamen statt neu berechnet zu werden (analog "cached input tokens" bei Anthropic/
OpenAI). Reports zeigen daher Input (prompt_tokens, gesamt), davon Cache (cache_tokens) und
Output (generation_tokens) getrennt.

  ./vllm_token_tracker.py collect                       # Dauerläufer (siehe systemd-Unit unten)
  ./vllm_token_tracker.py report --since 7d --group-by day,model
  ./vllm_token_tracker.py report --model qwen --since 2026-10-01 --until 2026-10-08 --format csv
  ./vllm_token_tracker.py report --session 12 --group-by hour
  ./vllm_token_tracker.py sessions                      # alle Server-Läufe mit Summen

Preise pro 1 Mio. Tokens (Input = Prompt ohne Cache-Treffer, Cache = Prompt aus dem Prefix-Cache,
Output = generiert), je Modell. Das Muster ist ein Teil des Modellnamens; passen mehrere, gewinnt
das längste, '*' gilt für alle übrigen Modelle. Sobald ein Preis hinterlegt ist, zeigen report und
sessions Kostenspalten. Gerechnet wird bei der Auswertung, eine Preisänderung wirkt also auch auf
alte Daten. --cache ist optional (Standard 0), falls der Anbieter Cache-Treffer nicht gesondert
abrechnet.
  ./vllm_token_tracker.py price set qwen3-32b --input 0.20 --cache 0.05 --output 0.60
  ./vllm_token_tracker.py price set '*' --input 0.10 --output 0.30
  ./vllm_token_tracker.py price list                    # Preise und Zuordnung zu den Modellen
  ./vllm_token_tracker.py price rm qwen3-32b

Strom: nvidia-smi misst nur die GPU. Was der Rest der Maschine zusätzlich zieht (CPU, RAM, SSD,
Netzteil), lässt sich als konstante Grundlast mit --overhead-watts oder VLLM_OVERHEAD_WATTS
einrechnen, am besten einmal mit einer Messsteckdose ermitteln. Abstände über 60 s zwischen zwei
Messwerten gelten als Messlücke (Collector oder Rechner aus) und zählen nicht zum Verbrauch.
  ./vllm_token_tracker.py power --since 7d --overhead-watts 25

systemd (als System-Dienst mit niedriger OOM-Priorität, damit er Speicherengpässe beim Laden
großer Modelle übersteht; %i = Benutzername):
  /etc/systemd/system/vllm-tokens@.service
    [Unit]
    Description=vLLM Token Tracker (%i)
    After=network.target docker.service
    [Service]
    User=%i
    ExecStart=/home/%i/bin/vllm_token_tracker.py collect
    Restart=always
    OOMScoreAdjust=-500
    [Install]
    WantedBy=multi-user.target
  sudo systemctl enable --now vllm-tokens@$USER
"""
import argparse
import bisect
import csv
import functools
import http.client
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

DB_DEFAULT = os.environ.get("VLLM_TOKEN_DB", str(Path.home() / ".local/share/vllm-tokens/tokens.db"))
OVERHEAD_DEFAULT = float(os.environ.get("VLLM_OVERHEAD_WATTS", "0"))
MAX_GAP = 60  # Sekunden; größere Abstände zwischen Messwerten sind Lücken (Collector/Rechner aus)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY,
    endpoint     TEXT NOT NULL,
    model        TEXT NOT NULL,
    server_start REAL,                    -- Startzeit des vLLM-Prozesses, falls /metrics sie liefert
    first_seen   REAL NOT NULL,
    last_seen    REAL NOT NULL,
    raw_prompt   REAL NOT NULL DEFAULT 0, -- letzter roher Zählerstand (für Deltas nach Collector-Neustart)
    raw_gen      REAL NOT NULL DEFAULT 0,
    raw_req      REAL NOT NULL DEFAULT 0,
    raw_cache    REAL NOT NULL DEFAULT 0  -- Teilmenge von raw_prompt: Cache-Treffer
);
CREATE TABLE IF NOT EXISTS samples (      -- Zuwachs pro Abfrage-Intervall
    ts                REAL NOT NULL,
    session_id        INTEGER NOT NULL REFERENCES sessions(id),
    prompt_tokens     INTEGER NOT NULL,
    generation_tokens INTEGER NOT NULL,
    requests          INTEGER NOT NULL,
    cache_tokens      INTEGER NOT NULL DEFAULT 0  -- Teilmenge von prompt_tokens: Cache-Treffer
);
CREATE INDEX IF NOT EXISTS samples_ts ON samples(ts);
CREATE TABLE IF NOT EXISTS prices (       -- Preis pro 1 Mio. Tokens
    pattern      TEXT PRIMARY KEY,        -- Teil des Modellnamens, '*' = alle übrigen
    input_per_m  REAL NOT NULL,
    output_per_m REAL NOT NULL,
    cache_per_m  REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS power_samples (  -- GPU-Leistungsaufnahme, per nvidia-smi abgefragt
    ts    REAL NOT NULL,
    watts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS power_samples_ts ON power_samples(ts);
"""

COUNTERS = {
    "vllm:prompt_tokens_total": "prompt",
    "vllm:prompt_tokens_cached_total": "cache",
    "vllm:generation_tokens_total": "gen",
    "vllm:request_success_total": "req",
}
LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
NET_ERRORS = (OSError, http.client.HTTPException, ValueError)


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")  # report kann lesen, während collect schreibt
    db.executescript(SCHEMA)
    migrate(db)
    return db


def migrate(db):
    """Spalten nachrüsten, die in älteren Datenbanken (vor Cache-Tracking) noch fehlen."""
    additions = (
        ("sessions", "raw_cache", "REAL NOT NULL DEFAULT 0"),
        ("samples", "cache_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("prices", "cache_per_m", "REAL NOT NULL DEFAULT 0"),
    )
    for table, column, decl in additions:
        cols = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    db.commit()


# ---------------------------------------------------------------- Collector

def scrape(endpoint, timeout=3.0):
    """Liest /metrics. Ergebnis: {modell: {prompt, gen, req, start}} oder None, wenn kein vLLM."""
    with urllib.request.urlopen(endpoint + "/metrics", timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")
    if "vllm:" not in text:
        return None
    models, proc_start = {}, None
    for line in text.splitlines():
        m = None if line.startswith("#") else LINE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), dict(LABEL.findall(m.group(2) or "")), float(m.group(3))
        if name == "process_start_time_seconds":
            proc_start = value
        elif name in COUNTERS or name == "vllm:generation_tokens_created":
            entry = models.setdefault(labels.get("model_name", "?"),
                                      {"prompt": 0.0, "gen": 0.0, "req": 0.0, "cache": 0.0, "start": None})
            if name in COUNTERS:  # über engine / finished_reason aufsummieren
                entry[COUNTERS[name]] += value
            else:
                entry["start"] = min(value, entry["start"] or value)
    for entry in models.values():
        entry["start"] = proc_start or entry["start"]
    return models


def sample_power():
    """Aktuelle Gesamt-Leistungsaufnahme aller GPUs in Watt, None falls nvidia-smi fehlt/fehlschlägt."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        watts = [float(line) for line in out.stdout.splitlines() if line.strip()]
        return sum(watts) if watts else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def power_rows(db, since=None, until=None):
    where, params = [], []
    if since is not None:
        where.append("ts >= ?"); params.append(since)
    if until is not None:
        where.append("ts < ?"); params.append(until)
    sql = "SELECT ts, watts FROM power_samples"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return db.execute(sql + " ORDER BY ts", params).fetchall()


def integrate(rows, overhead_watts=0.0):
    """(kWh, gemessene Stunden) per Trapez-Integration; Messlücken über MAX_GAP zählen nicht mit,
    sonst würde z. B. ein ausgeschalteter Rechner als Mittelwert der Nachbarwerte weiterverbrauchen."""
    wh = seconds = 0.0
    for (t0, w0), (t1, w1) in zip(rows, rows[1:]):
        if t1 - t0 <= MAX_GAP:
            wh += ((w0 + w1) / 2 + overhead_watts) * (t1 - t0) / 3600
            seconds += t1 - t0
    return wh / 1000, seconds / 3600


def energy_kwh(db, since=None, until=None, overhead_watts=0.0):
    """Energie in kWh und gemessene Stunden im Zeitraum."""
    return integrate(power_rows(db, since, until), overhead_watts)


def discover(host, min_port):
    """Alle lokalen Ports abklopfen, auf denen etwas lauscht, und vLLM-Server herausfiltern."""
    ports = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):  # lauschende Sockets direkt vom Kernel
        try:
            with open(table) as fh:
                for line in fh.readlines()[1:]:
                    fields = line.split()
                    if len(fields) > 3 and fields[3] == "0A":  # 0A = LISTEN
                        ports.add(int(fields[1].rsplit(":", 1)[1], 16))
        except (OSError, ValueError):
            continue
    try:  # zusätzlich veröffentlichte Container-Ports (falls Docker ohne Userland-Proxy läuft)
        out = subprocess.run(["docker", "ps", "--format", "{{.Ports}}"],
                             capture_output=True, text=True, timeout=5).stdout
        ports |= {int(p) for p in re.findall(r":(\d+)->", out)}
    except (OSError, subprocess.SubprocessError):
        pass

    def probe(port):
        endpoint = f"http://{host}:{port}"
        try:
            return endpoint if scrape(endpoint, timeout=1.0) is not None else None
        except NET_ERRORS:
            return None

    with ThreadPoolExecutor(max_workers=32) as pool:
        return {e for e in pool.map(probe, sorted(p for p in ports if p >= min_port)) if e}


INSERT_SAMPLE = ("INSERT INTO samples (ts, session_id, prompt_tokens, generation_tokens, requests, cache_tokens) "
                 "VALUES (?,?,?,?,?,?)")


def spread(db, sid, delta, start, end):
    """Nachgeholten Zuwachs gleichmäßig in Stundenschritten über [start, end] verteilen, damit er in
    Tages-/Stundenreports nicht als Spitze auf dem Zeitpunkt des Nachholens landet."""
    n = max(1, math.ceil((end - start) / 3600))
    for i in range(n):
        part = [d * (i + 1) // n - d * i // n for d in delta]  # Summe bleibt exakt delta
        if any(part):
            db.execute(INSERT_SAMPLE, (start + (end - start) * (i + 1) / n, sid, *part))


def record(db, endpoint, model, cur, now, started, verbose):
    """Zählerstand einer Session zuordnen und den Zuwachs speichern."""
    row = db.execute("SELECT id, server_start, raw_prompt, raw_gen, raw_req, raw_cache, last_seen FROM sessions "
                     "WHERE endpoint=? AND model=? ORDER BY id DESC LIMIT 1", (endpoint, model)).fetchone()
    same = False
    if row:
        if row[1] is not None and cur["start"] is not None:
            same = abs(row[1] - cur["start"]) < 2          # gleicher Prozess
        else:                                              # Fallback: Zähler nie rückläufig
            same = cur["prompt"] >= row[2] and cur["gen"] >= row[3] and cur["req"] >= row[4]
    if same:
        sid, base, previous = row[0], row[2:6], row[6]
    else:
        # Neuer Server-Lauf. Zähler starten bei Prozessstart mit 0, daher zählt der volle
        # aktuelle Stand - auch Tokens von vor dem Start des Collectors gehen so nicht verloren.
        sid = db.execute("INSERT INTO sessions (endpoint, model, server_start, first_seen, last_seen) "
                         "VALUES (?,?,?,?,?)", (endpoint, model, cur["start"], now, now)).lastrowid
        base = (0, 0, 0, 0)
        # lief der Server schon vor dem Collector, stammt der Stand aus der Zeit seit Serverstart
        previous = cur["start"] if cur["start"] is not None and cur["start"] < started else now
        print(f"[{datetime.now():%F %T}] neue Session #{sid}: {model} @ {endpoint}", flush=True)
    delta = [max(0, round(cur[k] - b)) for k, b in zip(("prompt", "gen", "req", "cache"), base)]
    if any(delta) and now - previous > MAX_GAP:
        spread(db, sid, delta, previous, now)
        print(f"  #{sid} {model}: Zuwachs seit {datetime.fromtimestamp(previous):%F %T} nachgeholt "
              f"und über {(now - previous) / 3600:.1f} h verteilt", flush=True)
    elif any(delta):
        db.execute(INSERT_SAMPLE, (now, sid, *delta))
        if verbose:
            print(f"  #{sid} {model}: +{delta[1]} generiert, +{delta[0]} Prompt (davon {delta[3]} Cache), "
                  f"+{delta[2]} Requests", flush=True)
    db.execute("UPDATE sessions SET last_seen=?, raw_prompt=?, raw_gen=?, raw_req=?, raw_cache=? WHERE id=?",
               (now, cur["prompt"], cur["gen"], cur["req"], cur["cache"], sid))


def cmd_collect(args):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    db = connect(args.db)
    fixed = {e.rstrip("/") for e in args.endpoint}
    endpoints, next_scan, started = set(fixed), 0.0, time.time()
    if args.interval * 2 > MAX_GAP:
        print(f"Warnung: --interval {args.interval:g} ist zu groß, Strom-Messwerte mit mehr als {MAX_GAP} s "
              "Abstand gelten als Lücke und zählen nicht.", file=sys.stderr, flush=True)
    print(f"Collector läuft, DB: {args.db}", flush=True)
    try:
        while True:
            now = time.time()
            if not args.no_scan and now >= next_scan:
                endpoints = fixed | discover(args.host, args.min_port)
                next_scan = now + args.rescan
            seen = set()  # derselbe Server unter zwei Adressen (z. B. --endpoint plus Scan) nur einmal zählen
            for endpoint in sorted(endpoints):
                try:
                    models = scrape(endpoint) or {}
                except NET_ERRORS:
                    continue  # Server gestoppt; taucht beim nächsten Scan nicht mehr auf
                for model, cur in models.items():
                    if cur["start"] is not None:
                        if (model, cur["start"]) in seen:
                            continue
                        seen.add((model, cur["start"]))
                    record(db, endpoint, model, cur, now, started, args.verbose)
            if not args.no_power:
                watts = sample_power()
                if watts is not None:
                    db.execute("INSERT INTO power_samples VALUES (?,?)", (now, watts))
            db.commit()
            time.sleep(max(0.0, args.interval - (time.time() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        db.commit()


# ---------------------------------------------------------------- Auswertung

GROUPS = {
    "hour": "strftime('%Y-%m-%d %H:00', s.ts, 'unixepoch', 'localtime')",
    "day": "date(s.ts, 'unixepoch', 'localtime')",
    "month": "strftime('%Y-%m', s.ts, 'unixepoch', 'localtime')",
    "model": "x.model",
    "endpoint": "x.endpoint",
    "session": "CAST(s.session_id AS TEXT)",
}
SUMS = ("prompt_tokens", "cache_tokens", "generation_tokens", "requests")
COSTS = ("kosten_input", "kosten_cache", "kosten_output", "kosten_gesamt")
COST_SQL = ["COALESCE(SUM((s.prompt_tokens - s.cache_tokens) * price_in(x.model)), 0) / 1e6 AS kosten_input",
            "COALESCE(SUM(s.cache_tokens * price_cache(x.model)), 0) / 1e6 AS kosten_cache",
            "COALESCE(SUM(s.generation_tokens * price_out(x.model)), 0) / 1e6 AS kosten_output"]


def price_lookup(db):
    """Liefert (Preise vorhanden?, Funktion Modell -> (Muster, Input-Preis, Output-Preis, Cache-Preis))."""
    prices = [tuple(r) for r in db.execute("SELECT pattern, input_per_m, output_per_m, cache_per_m FROM prices")]

    @functools.lru_cache(maxsize=None)
    def lookup(model):
        hits = [p for p in prices if p[0] != "*" and p[0].lower() in model.lower()]
        fallback = next((p for p in prices if p[0] == "*"), (None, 0.0, 0.0, 0.0))
        return max(hits, key=lambda p: len(p[0]), default=fallback)

    db.create_function("price_in", 1, lambda model: lookup(model)[1])
    db.create_function("price_out", 1, lambda model: lookup(model)[2])
    db.create_function("price_cache", 1, lambda model: lookup(model)[3])
    return bool(prices), lookup


def run_query(db, sql, params, lookup, priced):
    """Abfrage ausführen, Kosten runden und auf Modelle ohne Preis hinweisen."""
    rows = [dict(r) for r in db.execute(sql, params)]
    if priced:
        for row in rows:
            row["kosten_input"] = round(row["kosten_input"], 6)
            row["kosten_cache"] = round(row["kosten_cache"], 6)
            row["kosten_output"] = round(row["kosten_output"], 6)
            row["kosten_gesamt"] = round(row["kosten_input"] + row["kosten_cache"] + row["kosten_output"], 6)
        missing = sorted(m for (m,) in db.execute("SELECT DISTINCT model FROM sessions") if lookup(m)[0] is None)
        if missing:
            print(f"Hinweis: kein Preis hinterlegt (mit 0 gerechnet) für: {', '.join(missing)}", file=sys.stderr)
    return rows


def parse_time(value):
    """'90m', '24h', '7d', '2w', 'today' oder ISO-Datum/-Zeit (lokale Zeit)."""
    if value is None:
        return None
    rel = re.fullmatch(r"(\d+)([mhdw])", value)
    if rel:
        return time.time() - int(rel.group(1)) * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[rel.group(2)]
    if value == "today":
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return datetime.fromisoformat(value).timestamp()


def emit(rows, fmt, totals=True):
    if fmt == "json":
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        print("Keine Daten für diese Filter.", file=sys.stderr)
        return
    cols = list(rows[0])
    if fmt == "csv":
        writer = csv.DictWriter(sys.stdout, cols)
        writer.writeheader()
        writer.writerows(rows)
        return
    if totals and len(rows) > 1:
        rows = rows + [{c: sum(r[c] for r in rows) if c in SUMS + COSTS else ("SUMME" if c == cols[0] else "")
                        for c in cols}]
    swap = str.maketrans(",.", ".,")  # deutsche Schreibweise: 1.234,5678

    def cell(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:,.4f}".translate(swap)
        return f"{value:,}".translate(swap) if isinstance(value, int) else str(value)

    cells = [[cell(r[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(line[i]) for line in cells)) for i, c in enumerate(cols)]
    numeric = [isinstance(rows[0][c], (int, float)) for c in cols]
    fmt_line = lambda line: "  ".join(v.rjust(w) if n else v.ljust(w) for v, w, n in zip(line, widths, numeric))
    print(fmt_line(cols))
    print("  ".join("-" * w for w in widths))
    for i, line in enumerate(cells):
        if totals and len(rows) > 1 and i == len(cells) - 1:
            print("  ".join("-" * w for w in widths))
        print(fmt_line(line))


def cmd_report(args):
    groups = [g for g in args.group_by.split(",") if g and g != "none"]
    unknown = set(groups) - GROUPS.keys()
    if unknown:
        sys.exit(f"Unbekannte Gruppierung: {', '.join(unknown)} (möglich: {', '.join(GROUPS)}, none)")
    where, params = [], []
    for column, op, value in (("s.ts", ">=", parse_time(args.since)), ("s.ts", "<", parse_time(args.until)),
                              ("s.session_id", "=", args.session)):
        if value is not None:
            where.append(f"{column} {op} ?")
            params.append(value)
    for column, value in (("x.model", args.model), ("x.endpoint", args.endpoint)):
        if value:  # Teilstring, Groß-/Kleinschreibung egal
            where.append(f"instr(lower({column}), lower(?)) > 0")
            params.append(value)
    db = connect(args.db)
    db.row_factory = sqlite3.Row
    priced, lookup = price_lookup(db)
    select = [f"{GROUPS[g]} AS {g}" for g in groups] + [f"COALESCE(SUM(s.{c}), 0) AS {c}" for c in SUMS]
    select += COST_SQL if priced else []
    sql = f"SELECT {', '.join(select)} FROM samples s JOIN sessions x ON x.id = s.session_id"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if groups:
        positions = ", ".join(str(i + 1) for i in range(len(groups)))
        sql += f" GROUP BY {positions} ORDER BY {positions}"
    emit(run_query(db, sql, params, lookup, priced), args.format)


def cmd_sessions(args):
    db = connect(args.db)
    db.row_factory = sqlite3.Row
    priced, lookup = price_lookup(db)
    stamp = "strftime('%Y-%m-%d %H:%M', {}, 'unixepoch', 'localtime')"
    sql = (f"SELECT CAST(x.id AS TEXT) AS session, x.endpoint, x.model, "
           f"{stamp.format('COALESCE(x.server_start, x.first_seen)')} AS start, "
           f"{stamp.format('x.last_seen')} AS zuletzt_gesehen, "
           f"COALESCE(x.server_start, x.first_seen) AS t0, x.last_seen AS t1, "
           + ", ".join([f"COALESCE(SUM(s.{c}), 0) AS {c}" for c in SUMS] + (COST_SQL if priced else []))
           + " FROM sessions x LEFT JOIN samples s ON s.session_id = x.id GROUP BY x.id ORDER BY x.id")
    rows = run_query(db, sql, (), lookup, priced)
    # Strom im Zeitfenster jeder Session. Laufen Sessions parallel (z. B. Embedding neben LLM),
    # bekommt jede den vollen Verbrauch der GPU in ihrem Fenster.
    power = power_rows(db)
    stamps = [t for t, _ in power]
    for row in rows:
        t0, t1 = row.pop("t0"), row.pop("t1")
        kwh, hours = integrate(power[bisect.bisect_left(stamps, t0):bisect.bisect_right(stamps, t1)],
                               args.overhead_watts)
        row["kwh"] = round(kwh, 3)
        row["strom_abdeckung_pct"] = round(hours * 3600 / (t1 - t0) * 100) if t1 > t0 else None
        row["wh_pro_mio_output"] = (round(kwh * 1e9 / row["generation_tokens"], 1)
                                    if row["generation_tokens"] and hours else None)
    emit(rows, args.format, totals=False)


def cmd_power(args):
    db = connect(args.db)
    rows = power_rows(db, parse_time(args.since), parse_time(args.until))
    kwh, hours = integrate(rows, args.overhead_watts)
    if not hours:
        print("Keine Strom-Messwerte für diese Filter (Collector läuft noch nicht lange genug "
              "oder --no-power war gesetzt).", file=sys.stderr)
        return
    row = {"zeitraum_h": round((rows[-1][0] - rows[0][0]) / 3600, 2), "gemessen_h": round(hours, 2),
           "avg_watt": round(kwh * 1000 / hours, 1), "kwh": round(kwh, 4)}
    if args.overhead_watts:
        row["davon_grundlast_kwh"] = round(args.overhead_watts * hours / 1000, 4)
    row["kosten_euro"] = round(kwh * args.price, 4)
    emit([row], args.format, totals=False)


def cmd_roi(args):
    db = connect(args.db)
    db.row_factory = sqlite3.Row
    priced, lookup = price_lookup(db)
    if not priced:
        sys.exit("Keine Vergleichspreise hinterlegt, siehe 'price set'. Ohne Marktpreis keine ROI-Rechnung möglich.")
    since, until = parse_time(args.since), parse_time(args.until)
    where, params = [], []
    if since is not None:
        where.append("s.ts >= ?"); params.append(since)
    if until is not None:
        where.append("s.ts < ?"); params.append(until)
    select = [f"COALESCE(SUM(s.{c}), 0) AS {c}" for c in SUMS] + COST_SQL
    sql = f"SELECT {', '.join(select)} FROM samples s JOIN sessions x ON x.id = s.session_id"
    if where:
        sql += " WHERE " + " AND ".join(where)
    totals = run_query(db, sql, params, lookup, priced)[0]
    ersparnis = totals["kosten_gesamt"] * args.fx

    kwh, _ = energy_kwh(db, since, until, args.overhead_watts)
    stromkosten = kwh * args.kwh_price
    zeitkosten = args.hours * args.hourly_rate
    investition = args.hardware_cost + zeitkosten + stromkosten
    netto = ersparnis - investition

    # Raten über denselben Zeitraum wie Ersparnis und Strom, nicht pauschal ab der ersten Session
    first_seen = db.execute("SELECT MIN(first_seen) FROM sessions").fetchone()[0]
    start = max(since, first_seen) if since is not None and first_seen else (since or first_seen)
    end = min(until, time.time()) if until is not None else time.time()
    tage = max((end - start) / 86400, 0.001) if start else 0
    rate_pro_tag = ersparnis / tage if tage else 0
    strom_pro_tag = stromkosten / tage if tage else 0
    netto_pro_tag = rate_pro_tag - strom_pro_tag  # Strom läuft auch künftig weiter
    rest = investition - ersparnis

    row = {
        "ersparnis_bisher": round(ersparnis, 2),
        "stromkosten_bisher": round(stromkosten, 2),
        "kwh_bisher": round(kwh, 2),
        "zeitkosten": round(zeitkosten, 2),
        "hardware": round(args.hardware_cost, 2),
        "abschreibung_pro_jahr": round(args.hardware_cost / args.years, 2),
        "investition_gesamt": round(investition, 2),
        "netto_bisher": round(netto, 2),
        "ersparnis_pro_tag": round(rate_pro_tag, 2),
        "strom_pro_tag": round(strom_pro_tag, 2),
        "netto_pro_tag": round(netto_pro_tag, 2),
    }
    if rest > 0 and netto_pro_tag > 0:
        breakeven_tage = rest / netto_pro_tag
        row["noch_bis_breakeven_tage"] = round(breakeven_tage, 1)
        row["breakeven_ca"] = datetime.fromtimestamp(time.time() + breakeven_tage * 86400).strftime("%Y-%m-%d")
    elif rest <= 0:
        row["status"] = "break-even bereits erreicht"
    else:
        row["status"] = "Ersparnis deckt den Strom nicht, Prognose nicht möglich"
    emit([row], args.format, totals=False)


def cmd_price(args):
    db = connect(args.db)
    if args.action == "set":
        db.execute("INSERT OR REPLACE INTO prices VALUES (?,?,?,?)",
                   (args.pattern, args.input, args.output, args.cache))
        db.commit()
    elif args.action == "rm":
        if not db.execute("DELETE FROM prices WHERE pattern=?", (args.pattern,)).rowcount:
            sys.exit(f"Kein Preis mit Muster '{args.pattern}' vorhanden.")
        db.commit()
    priced, lookup = price_lookup(db)
    if not priced:
        print("Keine Preise hinterlegt. Beispiel: price set qwen --input 0.20 --output 0.60", file=sys.stderr)
        return
    rows = []
    for (model,) in db.execute("SELECT DISTINCT model FROM sessions ORDER BY model"):
        pattern, price_in, price_out, price_cache = lookup(model)
        rows.append({"modell": model, "muster": pattern or "(kein Preis)",
                     "input_pro_mio": float(price_in), "cache_pro_mio": float(price_cache),
                     "output_pro_mio": float(price_out)})
    used = {r["muster"] for r in rows}
    for pattern, price_in, price_out, price_cache in db.execute(
            "SELECT pattern, input_per_m, output_per_m, cache_per_m FROM prices ORDER BY pattern"):
        if pattern not in used:
            rows.append({"modell": "(ungenutzt)", "muster": pattern,
                         "input_pro_mio": price_in, "cache_pro_mio": price_cache, "output_pro_mio": price_out})
    emit(rows, args.format, totals=False)


def main():
    parser = argparse.ArgumentParser(description="Token-Tracking über alle vLLM-Sessions")
    parser.add_argument("--db", default=DB_DEFAULT, help=f"SQLite-Datei (Standard: {DB_DEFAULT})")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="Dauerläufer: vLLM-Instanzen finden und Zähler mitschreiben")
    c.add_argument("--interval", type=float, default=5, help="Sekunden zwischen Abfragen (Standard: 5)")
    c.add_argument("--rescan", type=float, default=30, help="Sekunden zwischen Suchläufen (Standard: 30)")
    c.add_argument("--endpoint", action="append", default=[], metavar="URL",
                   help="zusätzlicher fester Endpoint, z. B. http://192.168.1.20:8000 (mehrfach möglich)")
    c.add_argument("--no-scan", action="store_true", help="keine automatische Suche, nur --endpoint")
    c.add_argument("--no-power", action="store_true", help="keine GPU-Leistungsaufnahme per nvidia-smi erfassen")
    c.add_argument("--host", default="127.0.0.1", help="Host für die automatische Suche")
    c.add_argument("--min-port", type=int, default=1024, help="Ports darunter nicht abklopfen")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_collect)

    r = sub.add_parser("report", help="Tokens filtern und summieren")
    r.add_argument("--since", help="ab wann: 24h, 7d, today oder 2026-10-01[T08:00]")
    r.add_argument("--until", help="bis wann (exklusiv), gleiche Formate")
    r.add_argument("--model", help="Teil des Modellnamens")
    r.add_argument("--endpoint", help="Teil des Endpoints, z. B. :8001")
    r.add_argument("--session", type=int, help="Session-ID (siehe 'sessions')")
    r.add_argument("--group-by", default="day,model", help=f"Kommaliste aus {', '.join(GROUPS)} oder none")
    r.add_argument("--format", choices=("table", "csv", "json"), default="table")
    r.set_defaults(func=cmd_report)

    overhead = dict(type=float, default=OVERHEAD_DEFAULT,
                    help="Grundlast der restlichen Maschine in Watt, die nvidia-smi nicht sieht "
                         f"(Standard: VLLM_OVERHEAD_WATTS bzw. {OVERHEAD_DEFAULT:g})")

    s = sub.add_parser("sessions", help="alle Server-Läufe auflisten, mit Strom und Wh pro 1 Mio. Output-Tokens")
    s.add_argument("--overhead-watts", **overhead)
    s.add_argument("--format", choices=("table", "csv", "json"), default="table")
    s.set_defaults(func=cmd_sessions)

    pw = sub.add_parser("power", help="Stromverbrauch auswerten (nvidia-smi-Sampling aus 'collect')")
    pw.add_argument("--since", help="ab wann: 24h, 7d, today oder 2026-10-01[T08:00]")
    pw.add_argument("--until", help="bis wann (exklusiv), gleiche Formate")
    pw.add_argument("--price", type=float, default=0.34, help="Preis pro kWh in Euro (Standard: 0.34)")
    pw.add_argument("--overhead-watts", **overhead)
    pw.add_argument("--format", choices=("table", "csv", "json"), default="table")
    pw.set_defaults(func=cmd_power)

    roi = sub.add_parser("roi", help="ROI/Break-even ggü. öffentlichem API-Preis berechnen")
    roi.add_argument("--hours", type=float, required=True,
                     help="bisher investierte Zeit in Stunden (Setup + laufende Pflege)")
    roi.add_argument("--hourly-rate", type=float, default=95.0, help="Stundensatz in Euro (Standard: 95)")
    roi.add_argument("--hardware-cost", type=float, default=3850.0,
                     help="Anschaffungskosten der Hardware in Euro (Standard: 3850)")
    roi.add_argument("--years", type=float, default=3.0, help="Abschreibungszeitraum in Jahren (Standard: 3)")
    roi.add_argument("--kwh-price", type=float, default=0.34, help="Strompreis pro kWh in Euro (Standard: 0.34)")
    roi.add_argument("--overhead-watts", **overhead)
    roi.add_argument("--fx", type=float, default=1.0,
                     help="Kurs der Vergleichspreise in Euro, z. B. 0.92 bei Preisen in USD (Standard: 1)")
    roi.add_argument("--since", help="ab wann Ersparnis/Strom zählen (Standard: seit der ersten Session)")
    roi.add_argument("--until", help="bis wann (exklusiv)")
    roi.add_argument("--format", choices=("table", "csv", "json"), default="table")
    roi.set_defaults(func=cmd_roi)

    number = lambda text: float(text.replace(",", "."))  # 0,60 und 0.60 erlaubt
    p = sub.add_parser("price", help="Preise pro 1 Mio. Tokens je Modell verwalten")
    p.set_defaults(func=cmd_price)
    actions = p.add_subparsers(dest="action", required=True)
    a = actions.add_parser("set", help="Preis anlegen oder ändern")
    a.add_argument("pattern", metavar="MUSTER", help="Teil des Modellnamens, '*' für alle übrigen")
    a.add_argument("--input", type=number, required=True,
                   help="Preis pro 1 Mio. Input-Tokens (Prompt ohne Cache-Treffer)")
    a.add_argument("--output", type=number, required=True, help="Preis pro 1 Mio. Output-Tokens (generiert)")
    a.add_argument("--cache", type=number, default=0.0,
                   help="Preis pro 1 Mio. Cache-Tokens (Prefix-Cache-Treffer, meist günstiger; Standard: 0)")
    a = actions.add_parser("rm", help="Preis löschen")
    a.add_argument("pattern", metavar="MUSTER")
    actions.add_parser("list", help="Preise und ihre Zuordnung zu den erfassten Modellen")
    for action in actions.choices.values():
        action.add_argument("--format", choices=("table", "csv", "json"), default="table")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
