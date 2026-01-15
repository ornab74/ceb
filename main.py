"""
RGN CEB SYSTEM (Color-Entanglement Bits) — META-PROMPT GENERATOR ONLY
====================================================================

This file is intentionally a "meta-prompt generator":
- It does NOT embed actual domain countermeasures/runbooks/suggestions.
- It ONLY builds prompts that instruct an LLM to produce the suggestions/checklists.
- It keeps the CEB engine + entropy + memory + TUI and can call OpenAI via httpx.

Fix included:
- Robust psutil sampling in restricted environments where /proc/net/dev is unreadable
  (PermissionError). net counters fall back to 0 instead of crashing.

Install:
    pip install numpy psutil httpx

Optional env:
    export OPENAI_API_KEY="..."
    export OPENAI_MODEL="gpt-3.5-turbo"
    export OPENAI_BASE_URL="https://api.openai.com/v1"
    export RGN_TUI_REFRESH="0.75"

Run:
    python main.py

Keys:
    Q quit
    TAB cycle domain focus
    P toggle prompt preview
    O toggle AI output panel
    R force rescan
    A run AI for focused domain
    C toggle colorized view
"""

from __future__ import annotations

import os
import re
import time
import math
import json
import base64
import hashlib
import secrets
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import psutil
import curses
import httpx


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =============================================================================
# CONSTANTS
# =============================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

DEFAULT_DOMAINS = [
    "road_risk",
    "vehicle_security",
    "home_security",
    "medicine_compliance",
    "hygiene",
    "data_security",
]

DOMAIN_COUPLING = {
    "road_risk": ["vehicle_security"],
    "vehicle_security": ["road_risk", "data_security"],
    "home_security": ["data_security"],
    "medicine_compliance": ["hygiene"],
    "hygiene": ["medicine_compliance"],
    "data_security": ["vehicle_security", "home_security"],
}

TUI_REFRESH_SECONDS = float(os.environ.get("RGN_TUI_REFRESH", "0.75"))
ACTION_RE = re.compile(r"\[ACTION:(?P<cmd>[A-Z_]+)\s+(?P<args>.+?)\]", re.DOTALL)


# =============================================================================
# SYSTEM SIGNALS (robust in restricted procfs environments)
# =============================================================================
@dataclass
class SystemSignals:
    ram_used: int
    ram_total: int
    cpu_percent: float
    disk_percent: float
    net_sent: int
    net_recv: int
    uptime_s: float
    proc_count: int

    @staticmethod
    def sample() -> "SystemSignals":
        # Robust sampling: handle sandboxes/containers where /proc/* may be unreadable.
        vm_used = 0
        vm_total = 1
        cpu = 0.0
        disk = 0.0
        net_sent = 0
        net_recv = 0
        uptime = 0.0
        procs = 0

        try:
            vm = psutil.virtual_memory()
            vm_used = int(getattr(vm, "used", 0))
            vm_total = int(getattr(vm, "total", 1)) or 1
        except Exception:
            pass

        try:
            cpu = float(psutil.cpu_percent(interval=None))
        except Exception:
            pass

        try:
            disk = float(psutil.disk_usage("/").percent)
        except Exception:
            pass

        try:
            net = psutil.net_io_counters()
            net_sent = int(getattr(net, "bytes_sent", 0))
            net_recv = int(getattr(net, "bytes_recv", 0))
        except Exception:
            # PermissionError on /proc/net/dev is common in locked-down environments
            net_sent = 0
            net_recv = 0

        try:
            uptime = float(time.time() - psutil.boot_time())
        except Exception:
            uptime = 0.0

        try:
            procs = int(len(psutil.pids()))
        except Exception:
            procs = 0

        return SystemSignals(
            ram_used=vm_used,
            ram_total=vm_total,
            cpu_percent=cpu,
            disk_percent=disk,
            net_sent=net_sent,
            net_recv=net_recv,
            uptime_s=uptime,
            proc_count=procs,
        )


# =============================================================================
# RGB ENTROPY + LATTICE
# =============================================================================
def rgb_entropy_wheel(signals: SystemSignals) -> np.ndarray:
    t = time.perf_counter_ns()
    phase = (t ^ int(signals.cpu_percent * 1e6) ^ signals.ram_used ^ signals.net_sent ^ signals.net_recv) & 0xFFFFFFFF
    r = int((math.sin(phase * 1e-9) + 1.0) * 127.5) ^ secrets.randbits(8)
    g = int((math.sin(phase * 1e-9 + 2.09439510239) + 1.0) * 127.5) ^ secrets.randbits(8)
    b = int((math.sin(phase * 1e-9 + 4.18879020479) + 1.0) * 127.5) ^ secrets.randbits(8)
    return np.array([r & 0xFF, g & 0xFF, b & 0xFF], dtype=np.uint8)


def rgb_quantum_lattice(signals: SystemSignals) -> np.ndarray:
    """
    Byte-safe lattice fusion:
    - fuse base bytes with entropy RGB via add + xor (uint8)
    - convert to normalized float vector in [-1,1] then normalize
    """
    rgb = rgb_entropy_wheel(signals).astype(np.uint8)

    t = time.perf_counter_ns()
    cpu = signals.cpu_percent
    ram = signals.ram_used
    net = signals.net_sent ^ signals.net_recv

    base_u8 = np.array(
        [
            (t >> 0) & 0xFF, (t >> 8) & 0xFF, (t >> 16) & 0xFF, (t >> 24) & 0xFF,
            (t >> 32) & 0xFF, (t >> 40) & 0xFF, (t >> 48) & 0xFF, (t >> 56) & 0xFF,
            (net >> 0) & 0xFF, (net >> 8) & 0xFF,
            int(cpu * 10) & 0xFF,
            int((ram % 10_000_000) / 1000) & 0xFF,
        ],
        dtype=np.uint8,
    )

    fused = base_u8.copy()
    fused[0:3] = ((fused[0:3].astype(np.uint16) + rgb.astype(np.uint16)) % 256).astype(np.uint8)
    fused[3:6] = np.bitwise_xor(fused[3:6], rgb)

    fused_f = fused.astype(np.float64)
    v = (fused_f / 127.5) - 1.0
    v += np.random.normal(0.0, 0.03, size=v.shape)

    n = np.linalg.norm(v)
    if n < 1e-12:
        v[0] = 1.0
        n = 1.0
    return (v / n).astype(np.float64)


def amplify_entropy(signals: SystemSignals, lattice: np.ndarray) -> bytes:
    blob = lattice.tobytes()
    blob += secrets.token_bytes(96)
    blob += int(signals.ram_used).to_bytes(8, "little", signed=False)
    blob += int(signals.cpu_percent * 1000).to_bytes(8, "little", signed=False)
    blob += int(signals.disk_percent * 1000).to_bytes(8, "little", signed=False)
    blob += int(signals.net_sent).to_bytes(8, "little", signed=False)
    blob += int(signals.net_recv).to_bytes(8, "little", signed=False)
    return hashlib.sha3_512(blob).digest()


def shannon_entropy(prob: np.ndarray) -> float:
    p = np.clip(prob.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


# =============================================================================
# CEBs (Color-Entanglement Bits)
# =============================================================================
@dataclass
class CEBState:
    amps: np.ndarray
    colors: np.ndarray
    K: np.ndarray


class CEBEngine:
    def __init__(self, n_cebs: int = 24, seed: int = 0):
        self.n = int(n_cebs)
        self.rng = np.random.default_rng(seed if seed else None)

    def init_state(self, lattice: np.ndarray, seed_rgb: np.ndarray) -> CEBState:
        colors = np.zeros((self.n, 3), dtype=np.float64)
        sr = seed_rgb.astype(np.float64) / 255.0

        for i in range(self.n):
            base = sr + 0.15 * np.array(
                [
                    lattice[i % len(lattice)],
                    lattice[(i + 3) % len(lattice)],
                    lattice[(i + 7) % len(lattice)],
                ],
                dtype=np.float64,
            )
            colors[i] = np.mod(base, 1.0)

        amps = np.zeros((self.n,), dtype=np.complex128)
        for i in range(self.n):
            r, g, b = colors[i]
            hue_phase = (r * 0.9 + g * 1.3 + b * 0.7) * math.pi
            mag = 0.25 + 0.75 * abs(lattice[i % len(lattice)])
            amps[i] = mag * np.exp(1j * hue_phase)

        K = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dc = np.linalg.norm(colors[i] - colors[j])
                dl = abs(lattice[i % len(lattice)] - lattice[j % len(lattice)])
                w = math.exp(-3.0 * dc) * (0.4 + 0.6 * math.exp(-2.0 * dl))
                K[i, j] = w
                K[j, i] = w

        row_sums = np.sum(K, axis=1, keepdims=True) + 1e-12
        K = K / row_sums
        amps = amps / (np.linalg.norm(amps) + 1e-12)
        return CEBState(amps=amps, colors=colors, K=K)

    def evolve(
        self,
        st: CEBState,
        entropy_blob: bytes,
        steps: int = 180,
        drift_bias: float = 0.0,
        chroma_gain: float = 1.0,
    ) -> CEBState:
        drift_bias = float(np.clip(drift_bias, -0.75, 0.75))
        amps = st.amps.copy()
        colors = st.colors.copy()
        K = st.K

        ent = np.frombuffer(entropy_blob[:96], dtype=np.uint8).astype(np.float64) / 255.0
        D = np.diag(np.sum(K, axis=1))
        L = (D - K).astype(np.float64)

        for t in range(int(steps)):
            e = ent[t % len(ent)]
            phase_speed = (1.0 + 0.8 * abs(drift_bias))
            global_phase = np.exp(1j * (e * math.pi * phase_speed))

            shift = int(round(drift_bias * 5))
            coupled = K @ np.roll(amps, shift)

            grad_r = L @ colors[:, 0]
            grad_g = L @ colors[:, 1]
            grad_b = L @ colors[:, 2]
            grad_energy = np.sqrt(grad_r**2 + grad_g**2 + grad_b**2)
            grad_energy = grad_energy / (np.max(grad_energy) + 1e-12)

            nonlin = 0.35 + 0.65 * (grad_energy * chroma_gain)
            nonlin = np.clip(nonlin, 0.15, 1.25)

            amps = global_phase * (0.55 * amps + 0.45 * coupled) * nonlin
            amps = amps / (np.linalg.norm(amps) + 1e-12)

            rot = (0.002 + 0.004 * abs(drift_bias)) * (1.0 + 0.5 * e)
            colors = np.mod(colors + rot * np.array([1.0, 0.7, 0.4], dtype=np.float64), 1.0)

        adapt = 0.02 + 0.08 * max(0.0, drift_bias)
        if adapt > 0:
            newK = K.copy()
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    dc = np.linalg.norm(colors[i] - colors[j])
                    w = math.exp(-3.2 * dc)
                    newK[i, j] = (1 - adapt) * newK[i, j] + adapt * w
                    newK[j, i] = newK[i, j]
            row_sums = np.sum(newK, axis=1, keepdims=True) + 1e-12
            newK = newK / row_sums
        else:
            newK = K

        return CEBState(amps=amps, colors=colors, K=newK)

    def probs(self, st: CEBState) -> np.ndarray:
        p = np.abs(st.amps) ** 2
        p = p / (np.sum(p) + 1e-12)
        return p.astype(np.float64)

    def signature(self, st: CEBState, k: int = 12) -> Dict[str, Any]:
        p = self.probs(st)
        idx = np.argsort(p)[::-1][:k]
        top = []
        for i in idx:
            r, g, b = (st.colors[i] * 255.0).astype(int).tolist()
            top.append({"i": int(i), "p": float(p[i]), "rgb": [int(r), int(g), int(b)]})
        return {"entropy": float(shannon_entropy(p)), "top": top}


# =============================================================================
# MEMORY (per-domain entropy derived from domain slice)
# =============================================================================
class HierarchicalEntropicMemory:
    def __init__(self, short_n: int = 20, mid_n: int = 200, baseline_alpha: float = 0.005):
        self.short_n = int(short_n)
        self.mid_n = int(mid_n)
        self.baseline_alpha = float(baseline_alpha)
        self.short: Dict[str, List[float]] = {}
        self.mid: Dict[str, List[float]] = {}
        self.long: Dict[str, float] = {}

    def update(self, domain: str, entropy: float) -> None:
        self.short.setdefault(domain, []).append(float(entropy))
        self.mid.setdefault(domain, []).append(float(entropy))
        self.short[domain] = self.short[domain][-self.short_n:]
        self.mid[domain] = self.mid[domain][-self.mid_n:]
        if domain not in self.long:
            self.long[domain] = float(entropy)
        else:
            b = self.long[domain]
            a = self.baseline_alpha
            self.long[domain] = (1.0 - a) * b + a * float(entropy)

    def stats(self, domain: str) -> Dict[str, float]:
        s = self.short.get(domain, [])
        m = self.mid.get(domain, [])
        baseline = self.long.get(domain, float(np.mean(s)) if s else 0.0)
        short_mean = float(np.mean(s)) if s else 0.0
        mid_mean = float(np.mean(m)) if m else 0.0
        short_var = float(np.var(s)) if len(s) >= 2 else 0.0
        mid_var = float(np.var(m)) if len(m) >= 2 else 0.0
        volatility = short_var + 0.5 * mid_var
        return {"short_mean": short_mean, "mid_mean": mid_mean, "baseline": float(baseline), "volatility": float(volatility)}

    def drift(self, domain: str) -> float:
        st = self.stats(domain)
        return float(st["short_mean"] - st["baseline"])

    def confidence(self, domain: str) -> float:
        st = self.stats(domain)
        conf = 1.0 / (1.0 + st["volatility"])
        return float(max(0.1, min(0.99, conf)))


# =============================================================================
# DOMAIN SLICE + RISK (scaled)
# =============================================================================
def _domain_slice(domain: str, p: np.ndarray) -> np.ndarray:
    n = len(p)
    a = max(1, n // 6)
    if domain == "road_risk":
        return p[0:a]
    if domain == "vehicle_security":
        return p[a:2 * a]
    if domain == "home_security":
        return p[2 * a:3 * a]
    if domain == "medicine_compliance":
        return p[3 * a:4 * a]
    if domain == "hygiene":
        return p[4 * a:5 * a]
    if domain == "data_security":
        return p[5 * a:]
    return p


def domain_entropy_from_slice(sl: np.ndarray) -> float:
    sln = sl / (np.sum(sl) + 1e-12)
    return shannon_entropy(sln)


def domain_risk_from_ceb(domain: str, p: np.ndarray) -> float:
    """
    Uses slice mass relative to uniform expected mass, then maps to 0..1.
    This is a *dial* for prompt conditioning, not a real-world risk claim.
    """
    sl = _domain_slice(domain, p)
    mass = float(np.sum(sl))
    expected = len(sl) / max(1.0, float(len(p)))
    scaled = mass / (expected + 1e-12)   # ~1 at uniform

    r = (scaled - 0.8) / 1.6
    return float(np.clip(r, 0.0, 1.0))


def apply_cross_domain_bias(domain: str, base_risk: float, memory: HierarchicalEntropicMemory) -> float:
    bias = 0.0
    for linked in DOMAIN_COUPLING.get(domain, []):
        d = memory.drift(linked)
        if d > 0:
            bias += min(0.12, d * 0.06)
    return float(np.clip(base_risk + bias, 0.0, 1.0))


def status_from_risk(r: float) -> str:
    if r < 0.33:
        return "LOW"
    if r < 0.66:
        return "MODERATE"
    return "HIGH"


# =============================================================================
# PROMPT CHUNKS (META-PROMPT ONLY) + ACTIONS (base64 hardened)
# =============================================================================
@dataclass
class PromptChunk:
    id: str
    title: str
    text: str
    rgb: Tuple[int, int, int]
    weight: float
    pos: int

    def as_text(self, with_rgb_tags: bool = True) -> str:
        r, g, b = self.rgb
        if with_rgb_tags:
            return f"<RGB {r},{g},{b} CHUNK={self.id} POS={self.pos} W={self.weight:.6f}>\n[{self.title}]\n{self.text}\n</RGB>\n"
        return f"[{self.title}]\n{self.text}\n"


@dataclass
class PromptDraft:
    chunks: List[PromptChunk]
    temperature: float = 0.5
    max_tokens: int = 512
    notes: List[str] = field(default_factory=list)

    def render(self, with_rgb_tags: bool = True) -> str:
        return "\n".join(c.as_text(with_rgb_tags=with_rgb_tags) for c in self.chunks).strip() + "\n"


def parse_kv_args(argstr: str) -> Dict[str, str]:
    pattern = re.compile(r'(\w+)=(".*?"|\'.*?\'|\S+)')
    out: Dict[str, str] = {}
    for k, v in pattern.findall(argstr):
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k] = v
    return out


def encode_b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("utf-8")


def decode_text_arg(args: Dict[str, str]) -> str:
    if "text_b64" in args:
        try:
            return base64.b64decode(args["text_b64"].encode("utf-8")).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return args.get("text", "")


def apply_actions(draft: PromptDraft, action_text: str) -> None:
    for m in ACTION_RE.finditer(action_text):
        cmd = m.group("cmd").strip().upper()
        args = parse_kv_args(m.group("args").strip())

        if cmd == "SET_TEMPERATURE":
            try:
                val = float(args.get("value", "0.5"))
                draft.temperature = float(max(0.0, min(1.5, val)))
                draft.notes.append(f"temp={draft.temperature}")
            except Exception:
                pass

        elif cmd == "SET_MAX_TOKENS":
            try:
                val = int(float(args.get("value", "512")))
                draft.max_tokens = int(max(64, min(2048, val)))
                draft.notes.append(f"max_tokens={draft.max_tokens}")
            except Exception:
                pass

        elif cmd == "ADD_SECTION":
            title = args.get("title", "NEW_SECTION")
            text = decode_text_arg(args)
            h = hashlib.sha256((title + text).encode("utf-8")).digest()
            rgb = (int(h[0]), int(h[1]), int(h[2]))
            draft.chunks.append(
                PromptChunk(
                    id=f"ADD_{len(draft.chunks):02d}",
                    title=title,
                    text=text,
                    rgb=rgb,
                    weight=0.01,
                    pos=len(draft.chunks),
                )
            )
            draft.notes.append(f"add={title}")

        elif cmd == "REWRITE_SECTION":
            title = args.get("title", "")
            text = decode_text_arg(args)
            for c in draft.chunks:
                if c.title == title:
                    c.text = text
                    draft.notes.append(f"rewrite={title}")
                    break

        elif cmd == "PRIORITIZE":
            sections = [s.strip() for s in args.get("sections", "").split(",") if s.strip()]
            if not sections:
                continue
            header = [c for c in draft.chunks if c.title == "SYSTEM_HEADER"]
            rest = [c for c in draft.chunks if c.title != "SYSTEM_HEADER"]
            title_to_chunk = {c.title: c for c in rest}
            prioritized = [title_to_chunk[t] for t in sections if t in title_to_chunk]
            remaining = [c for c in rest if c.title not in set(sections)]
            draft.chunks = header + prioritized + remaining
            for i, c in enumerate(draft.chunks):
                c.pos = i
            draft.notes.append("prioritize=" + ",".join(sections))

        elif cmd == "TRIM":
            try:
                max_chars = int(float(args.get("max_chars", "20000")))
            except Exception:
                max_chars = 20000

            keep = {"SYSTEM_HEADER", "OUTPUT_SCHEMA", "NONNEGOTIABLE_RULES", "DOMAIN_SPEC"}
            drop = [c for c in draft.chunks if c.title not in keep]
            drop.sort(key=lambda c: c.weight)

            def length_now(chs: List[PromptChunk]) -> int:
                return len("\n".join(c.as_text(True) for c in chs))

            chs = draft.chunks[:]
            while length_now(chs) > max_chars and drop:
                victim = drop.pop(0)
                chs = [c for c in chs if c is not victim]
            draft.chunks = chs
            for i, c in enumerate(draft.chunks):
                c.pos = i
            draft.notes.append(f"trim={max_chars}")


# =============================================================================
# META-PROMPT CONTENT (NO embedded advice)
# =============================================================================
def build_output_schema() -> str:
    return "\n".join([
        "OUTPUT FORMAT (must follow exactly):",
        "",
        "SUMMARY:",
        "- 1–2 lines. No fluff.",
        "",
        "ASSUMPTIONS:",
        "- Bullet list of assumptions made due to missing data.",
        "",
        "QUESTIONS_FOR_USER:",
        "- 3–7 short questions to request missing inputs (only what’s necessary).",
        "",
        "FINDINGS:",
        "- Bullet list. Each bullet: (signal → inference → impact).",
        "",
        "ACTIONS_BY_TIME_WINDOW:",
        "A) 0–30 min:",
        "- Action: ...",
        "  Why: ...",
        "  Verification: ...",
        "  NextCheck: ...",
        "",
        "B) 2 hours:",
        "- Action: ...",
        "  Why: ...",
        "  Verification: ...",
        "  NextCheck: ...",
        "",
        "C) 2–12 hours:",
        "- Action: ...",
        "  Why: ...",
        "  Verification: ...",
        "  NextCheck: ...",
        "",
        "D) 12–48 hours:",
        "- Action: ...",
        "  Why: ...",
        "  Verification: ...",
        "  NextCheck: ...",
        "",
        "ALERTS:",
        "- Only if risk is HIGH; concise, explicit triggers and what to do.",
        "",
        "SAFETY_NOTES:",
        "- Any relevant boundary notes (e.g., consult a professional when appropriate).",
    ])


def build_nonnegotiable_rules() -> str:
    return "\n".join([
        "NONNEGOTIABLE RULES:",
        "- Do not claim you have sensors, external data, or certainty.",
        "- If data is missing: state assumptions + ask targeted questions + still give a lowest-regret plan.",
        "- Avoid fear-mongering; be calm and operational.",
        "- Do not provide illegal instructions.",
        "- Use measurable verification steps and explicit next-check timing.",
        "- Keep actions practical and clearly sequenced.",
    ])


def build_domain_spec(domain: str) -> str:
    common = [
        "You are generating an operational checklist and plan for the specified domain.",
        "You must adapt to the user's context if provided in USER_CONTEXT.",
        "If USER_CONTEXT is empty, provide a generic plan and ask questions.",
        "Avoid domain-specific 'libraries' of steps unless justified by user context; keep it minimal and safe.",
    ]

    if domain == "road_risk":
        domain_lines = [
            "DOMAIN SPEC: ROAD_RISK",
            "- Produce 3 SAFE WINDOWS and 2 AVOID WINDOWS for the next 48 hours based on USER_CONTEXT.",
            "- Evaluate: driver readiness, route conditions, vehicle readiness, and timing constraints.",
            "- Include a verification method for each time window (what to check / confirm).",
            "- Ask for missing: departure time, route, weather snapshot, sleep/fatigue indicators, vehicle state notes.",
        ]
    elif domain == "vehicle_security":
        domain_lines = [
            "DOMAIN SPEC: VEHICLE_SECURITY",
            "- Produce a 2–5 minute quick-check plan and a 15–30 minute deeper-check plan.",
            "- Identify likely threat categories only as hypotheses; do not assert compromise.",
            "- Ask for missing: last known secure time, recent service, parking pattern, key behavior anomalies, vehicle make/model/year.",
            "- Include verification observations and what each observation would imply.",
        ]
    elif domain == "home_security":
        domain_lines = [
            "DOMAIN SPEC: HOME_SECURITY",
            "- Separate physical perimeter and digital perimeter into distinct sections.",
            "- Produce a '10-minute minimum hardening' plan and an extended plan.",
            "- Ask for missing: router model, device inventory, existing security devices, recent visitors/contractors.",
            "- Include verification: account/device audits and how to confirm changes took effect.",
        ]
    elif domain == "medicine_compliance":
        domain_lines = [
            "DOMAIN SPEC: MEDICINE_COMPLIANCE",
            "- Do NOT change dosage or provide medical directives beyond adherence support.",
            "- Produce reminder/anchor strategies and a plan to reduce missed doses.",
            "- Ask for missing: medication schedule, constraints (food, time windows), routine anchors, refill status.",
            "- Include verification steps like confirmation logging or check-off methods.",
        ]
    elif domain == "hygiene":
        domain_lines = [
            "DOMAIN SPEC: HYGIENE",
            "- Produce a minimal routine plus optional upgrades, tied to triggers and schedule.",
            "- Ask for missing: exposure context, current routine, supply constraints, time availability.",
            "- Include verification: how to measure compliance (simple tracking).",
        ]
    elif domain == "data_security":
        domain_lines = [
            "DOMAIN SPEC: DATA_SECURITY",
            "- Produce a safe triage plan: assess, contain, verify, recover.",
            "- Do not provide malware creation or illegal hacking instructions.",
            "- Ask for missing: OS/device type, recent alerts, suspicious events, key accounts, backup status.",
            "- Include verification steps: how to confirm accounts/devices are secured after actions.",
        ]
    else:
        domain_lines = [f"DOMAIN SPEC: {domain}", "- Produce an operational plan and ask for missing context."]

    return "\n".join(common + [""] + domain_lines)


def build_user_context_placeholder(domain: str) -> str:
    return "\n".join([
        "USER_CONTEXT (fill this in; if empty, ask questions):",
        f"- domain={domain}",
        "- goal:",
        "- constraints:",
        "- timeline:",
        "- relevant details:",
        "- what happened / what you observed:",
        "- what you already tried:",
    ])


# =============================================================================
# CEB-BASED CHUNKER (builds meta-prompt)
# =============================================================================
class CEBChunker:
    def __init__(self, max_chunks: int = 14):
        self.max_chunks = int(max_chunks)

    def build(self, domain: str, metrics: Dict[str, float], ceb_sig: Dict[str, Any], base_rgb: np.ndarray) -> List[PromptChunk]:
        top = ceb_sig.get("top", [])
        ent = float(ceb_sig.get("entropy", 0.0))

        def pick_rgb(i: int) -> Tuple[int, int, int]:
            if top:
                rgb = np.array(top[i % len(top)]["rgb"], dtype=np.int32)
            else:
                rgb = base_rgb.astype(np.int32)
            mix = (0.65 * rgb + 0.35 * base_rgb.astype(np.int32))
            mix = np.clip(mix, 0, 255).astype(int)
            return int(mix[0]), int(mix[1]), int(mix[2])

        risk = float(metrics["risk"])
        drift = float(metrics["drift"])
        conf = float(metrics["confidence"])
        vol = float(metrics["volatility"])

        chunks: List[Tuple[str, str, float]] = [
            ("SYSTEM_HEADER", f"RGN-CEB META-PROMPT GENERATOR\nDOMAIN={domain}\n", 10.0),
            ("STATE_METRICS", "\n".join([
                "NOTE: metrics are internal dials, not real-world measurements.",
                f"risk_dial={risk:.4f}",
                f"status_band={status_from_risk(risk)}",
                f"drift={drift:+.4f}",
                f"confidence={conf:.4f}",
                f"volatility={vol:.6f}",
                f"ceb_entropy={ent:.4f}",
            ]), 9.2),
            ("CEB_SIGNATURE", json.dumps(ceb_sig, ensure_ascii=False, indent=2), 8.4),
            ("NONNEGOTIABLE_RULES", build_nonnegotiable_rules(), 9.0),
            ("DOMAIN_SPEC", build_domain_spec(domain), 8.8),
            ("USER_CONTEXT", build_user_context_placeholder(domain), 8.6),
            ("OUTPUT_SCHEMA", build_output_schema(), 9.1),
            ("QUALITY_GATES", "\n".join([
                "QUALITY GATES (enforce):",
                "- Every time window must include: Action + Why + Verification + NextCheck.",
                "- If confidence < 0.65: ask more questions and choose conservative actions.",
                "- If status_band is HIGH: include immediate containment-oriented actions and clear alert triggers.",
                "- Keep it actionable; avoid long explanations.",
            ]), 7.5),
        ]

        out: List[PromptChunk] = []
        for i, (title, txt, base_w) in enumerate(chunks[: self.max_chunks]):
            rgb = pick_rgb(i)
            w = base_w * (1.0 + 0.55 * risk + 0.25 * abs(drift)) * (0.85 + 0.15 * conf)
            out.append(PromptChunk(
                id=f"{domain[:4].upper()}_{i:02d}",
                title=title,
                text=txt,
                rgb=rgb,
                weight=float(w),
                pos=i,
            ))

        header = [c for c in out if c.title == "SYSTEM_HEADER"]
        rest = [c for c in out if c.title != "SYSTEM_HEADER"]
        rest.sort(key=lambda c: c.weight, reverse=True)
        ordered = header + rest
        for i, c in enumerate(ordered):
            c.pos = i
        return ordered


# =============================================================================
# SUB-AGENTS (meta-prompt mutation only)
# =============================================================================
class SubPromptAgent:
    name: str = "base"
    def propose_actions(self, ctx: Dict[str, Any], draft: PromptDraft) -> str:
        return ""


class HardenerAgent(SubPromptAgent):
    name = "hardener"
    def propose_actions(self, ctx: Dict[str, Any], draft: PromptDraft) -> str:
        risk = float(ctx["risk"])
        conf = float(ctx["confidence"])

        extra = []
        if risk >= 0.66:
            extra += [
                "- HIGH band: require alert triggers in ALERTS section and require an immediate (0–30 min) block.",
                "- HIGH band: require a containment-first ordering (contain → verify → recover).",
            ]
        if conf < 0.65:
            extra += [
                "- Low confidence: QUESTIONS_FOR_USER must be prioritized and limited to the most informative items.",
                "- Low confidence: actions must be reversible and low-regret.",
            ]

        if not extra:
            return ""

        text = build_nonnegotiable_rules() + "\n" + "\n".join(extra)
        return f'[ACTION:REWRITE_SECTION title="NONNEGOTIABLE_RULES" text_b64="{encode_b64(text)}"]'


class PrioritizerAgent(SubPromptAgent):
    name = "prioritizer"
    def propose_actions(self, ctx: Dict[str, Any], draft: PromptDraft) -> str:
        risk = float(ctx["risk"])
        conf = float(ctx["confidence"])
        if risk >= 0.66:
            return "[ACTION:PRIORITIZE sections=STATE_METRICS,DOMAIN_SPEC,USER_CONTEXT,OUTPUT_SCHEMA,NONNEGOTIABLE_RULES,QUALITY_GATES,CEB_SIGNATURE]"
        if conf < 0.65:
            return "[ACTION:PRIORITIZE sections=STATE_METRICS,USER_CONTEXT,NONNEGOTIABLE_RULES,DOMAIN_SPEC,OUTPUT_SCHEMA,QUALITY_GATES]"
        return "[ACTION:PRIORITIZE sections=STATE_METRICS,DOMAIN_SPEC,USER_CONTEXT,OUTPUT_SCHEMA,NONNEGOTIABLE_RULES]"


class TempTokenAgent(SubPromptAgent):
    name = "temp_token"
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 4))

    def propose_actions(self, ctx: Dict[str, Any], draft: PromptDraft) -> str:
        risk = float(ctx["risk"])
        conf = float(ctx["confidence"])
        vol = float(ctx["volatility"])
        drift = float(ctx["drift"])

        base_temp = 0.35 + 0.22 * (1.0 - risk)
        base_temp -= 0.14 * (1.0 - conf)
        base_temp -= 0.10 * min(1.0, vol)
        base_temp -= 0.08 * min(1.0, abs(drift))
        temp = float(max(0.06, min(0.75, base_temp)))

        est_in = self._estimate_tokens(draft.render(with_rgb_tags=True))
        if est_in > 2600:
            out = 256
        elif est_in > 1900:
            out = 384
        else:
            out = 500 + int(450 * min(1.0, risk + (1.0 - conf)))
            out = min(1100, max(256, out))

        return f"[ACTION:SET_TEMPERATURE value={temp}] [ACTION:SET_MAX_TOKENS value={out}]"


class LengthGuardAgent(SubPromptAgent):
    name = "length_guard"
    def propose_actions(self, ctx: Dict[str, Any], draft: PromptDraft) -> str:
        max_chars = int(ctx.get("max_prompt_chars", 22000))
        if len(draft.render(True)) > max_chars:
            return f"[ACTION:TRIM max_chars={max_chars}]"
        return ""


# =============================================================================
# ORCHESTRATOR
# =============================================================================
@dataclass
class PromptPlan:
    domain: str
    prompt: str
    temperature: float
    max_tokens: int
    meta: Dict[str, Any] = field(default_factory=dict)


class PromptOrchestrator:
    def __init__(self):
        self.chunker = CEBChunker(max_chunks=14)
        self.agents: List[SubPromptAgent] = [
            HardenerAgent(),
            PrioritizerAgent(),
            TempTokenAgent(),
            LengthGuardAgent(),
        ]

    def build_plan(
        self,
        domain: str,
        metrics: Dict[str, float],
        ceb_sig: Dict[str, Any],
        base_rgb: np.ndarray,
        max_prompt_chars: int = 22000,
        with_rgb_tags: bool = True
    ) -> PromptPlan:
        chunks = self.chunker.build(domain=domain, metrics=metrics, ceb_sig=ceb_sig, base_rgb=base_rgb)
        draft = PromptDraft(chunks=chunks, temperature=0.35, max_tokens=512)

        ctx = dict(metrics)
        ctx["domain"] = domain
        ctx["max_prompt_chars"] = max_prompt_chars

        for agent in self.agents:
            actions = agent.propose_actions(ctx, draft)
            if actions:
                apply_actions(draft, actions)

        prompt = draft.render(with_rgb_tags=with_rgb_tags)
        meta = {
            "notes": draft.notes,
            "chars": len(prompt),
            "ceb_entropy": float(ceb_sig.get("entropy", 0.0)),
            "top_colors": [t["rgb"] for t in ceb_sig.get("top", [])[:6]],
        }
        return PromptPlan(domain=domain, prompt=prompt, temperature=draft.temperature, max_tokens=draft.max_tokens, meta=meta)


# =============================================================================
# httpx OpenAI client
# =============================================================================
class HttpxOpenAIClient:
    def __init__(self, api_key: str, base_url: str = OPENAI_BASE_URL, model: str = OPENAI_MODEL, timeout_s: float = 60.0):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(timeout_s)

    def chat(self, prompt: str, temperature: float, max_tokens: int, retries: int = 3) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        last_err: Optional[Exception] = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(int(retries)):
                try:
                    r = client.post(url, headers=headers, json=payload)
                    if r.status_code >= 400:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                    j = r.json()
                    return j["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    last_err = e
                    time.sleep((2 ** attempt) * 0.6)
        raise RuntimeError(f"OpenAI call failed: {last_err}")


# =============================================================================
# SCAN RESULT STRUCT
# =============================================================================
@dataclass
class DomainScan:
    domain: str
    risk: float
    drift: float
    confidence: float
    volatility: float
    status: str
    ceb_entropy: float
    ceb_top: List[Dict[str, Any]]
    prompt_plan: Optional[PromptPlan] = None
    last_ai_output: str = ""


# =============================================================================
# CORE SYSTEM
# =============================================================================
class RGNCebSystem:
    def __init__(self, domains: Optional[List[str]] = None):
        self.domains = domains or list(DEFAULT_DOMAINS)
        self.memory = HierarchicalEntropicMemory()
        self.ceb = CEBEngine(n_cebs=24)
        self.orch = PromptOrchestrator()

        self.last_scans: Dict[str, DomainScan] = {}
        self.focus_idx = 0
        self._lock = threading.Lock()

    def scan_once(self) -> Dict[str, DomainScan]:
        signals = SystemSignals.sample()
        lattice = rgb_quantum_lattice(signals)
        ent_blob = amplify_entropy(signals, lattice)
        base_rgb = rgb_entropy_wheel(signals)

        st0 = self.ceb.init_state(lattice=lattice, seed_rgb=base_rgb)

        global_bias = 0.0
        with self._lock:
            if self.last_scans:
                global_bias = float(np.mean([s.drift for s in self.last_scans.values()]))

        st = self.ceb.evolve(st0, entropy_blob=ent_blob, steps=180, drift_bias=global_bias, chroma_gain=1.15)
        p = self.ceb.probs(st)
        sig = self.ceb.signature(st, k=12)

        scans: Dict[str, DomainScan] = {}

        for d in self.domains:
            sl = _domain_slice(d, p)
            d_entropy = domain_entropy_from_slice(sl)
            self.memory.update(d, d_entropy)

            drift = self.memory.drift(d)
            conf = self.memory.confidence(d)
            vol = self.memory.stats(d)["volatility"]

            base_risk = domain_risk_from_ceb(d, p)
            risk = apply_cross_domain_bias(d, base_risk, self.memory)
            status = status_from_risk(risk)

            metrics = {"risk": float(risk), "drift": float(drift), "confidence": float(conf), "volatility": float(vol)}

            plan = self.orch.build_plan(
                domain=d,
                metrics=metrics,
                ceb_sig=sig,
                base_rgb=base_rgb,
                max_prompt_chars=22000,
                with_rgb_tags=True,
            )

            prev_out = ""
            with self._lock:
                if d in self.last_scans:
                    prev_out = self.last_scans[d].last_ai_output

            scans[d] = DomainScan(
                domain=d,
                risk=float(risk),
                drift=float(drift),
                confidence=float(conf),
                volatility=float(vol),
                status=status,
                ceb_entropy=float(sig.get("entropy", 0.0)),
                ceb_top=sig.get("top", []),
                prompt_plan=plan,
                last_ai_output=prev_out,
            )

        with self._lock:
            self.last_scans = scans
        return scans

    def get_focus_domain(self) -> str:
        return self.domains[self.focus_idx % len(self.domains)]

    def cycle_focus(self) -> None:
        self.focus_idx = (self.focus_idx + 1) % len(self.domains)


# =============================================================================
# CURSES COLOR HELPERS
# =============================================================================
def rgb_to_xterm256(r: int, g: int, b: int) -> int:
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + int((r - 8) / 10)

    def to_6(v: int) -> int:
        return int(round((v / 255) * 5))

    ri, gi, bi = to_6(r), to_6(g), to_6(b)
    return 16 + 36 * ri + 6 * gi + bi


class ColorPairCache:
    def __init__(self, max_pairs: int = 72):
        self.max_pairs = max_pairs
        self._lru: "OrderedDict[int, int]" = OrderedDict()
        self._next_pair_id = 1

    def get_pair(self, color_index: int) -> int:
        if color_index in self._lru:
            pair_id = self._lru.pop(color_index)
            self._lru[color_index] = pair_id
            return pair_id

        if len(self._lru) >= self.max_pairs:
            _, evicted_pair = self._lru.popitem(last=False)
            pair_id = evicted_pair
        else:
            pair_id = self._next_pair_id
            self._next_pair_id += 1

        try:
            curses.init_pair(pair_id, color_index, -1)
        except Exception:
            pair_id = 0

        self._lru[color_index] = pair_id
        return pair_id


# =============================================================================
# TUI
# =============================================================================
class AdvancedTUI:
    def __init__(self, system: RGNCebSystem):
        self.sys = system
        self.scans: Dict[str, DomainScan] = {}
        self.show_prompt = True
        self.show_ai_output = False
        self.colorized_prompt = True
        self.logs: List[str] = []
        self.last_refresh = 0.0
        self._lock = threading.Lock()
        self._color_cache = ColorPairCache(max_pairs=72)

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"{ts} {msg}"
        with self._lock:
            self.logs.append(line)
            self.logs = self.logs[-160:]

    def run(self) -> None:
        curses.wrapper(self._main)

    def _main(self, stdscr) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)

        try:
            curses.start_color()
            curses.use_default_colors()
        except Exception:
            pass

        self.log("TUI started. Q quit | TAB domain | P prompt | O output | C color | R rescan | A AI")

        with self._lock:
            self.scans = self.sys.scan_once()
        self.log("Initial scan complete.")

        while True:
            now = time.time()
            if now - self.last_refresh > TUI_REFRESH_SECONDS:
                scans = self.sys.scan_once()
                with self._lock:
                    self.scans = scans
                self.last_refresh = now

            self._draw(stdscr)

            ch = stdscr.getch()
            if ch == -1:
                continue

            if ch in (ord("q"), ord("Q")):
                self.log("Quit.")
                break
            if ch == 9:  # TAB
                self.sys.cycle_focus()
                self.log(f"Focus domain: {self.sys.get_focus_domain()}")
            elif ch in (ord("p"), ord("P")):
                self.show_prompt = not self.show_prompt
                self.log(f"Prompt preview: {'ON' if self.show_prompt else 'OFF'}")
            elif ch in (ord("o"), ord("O")):
                self.show_ai_output = not self.show_ai_output
                self.log(f"AI output panel: {'ON' if self.show_ai_output else 'OFF'}")
            elif ch in (ord("c"), ord("C")):
                self.colorized_prompt = not self.colorized_prompt
                self.log(f"Colorized prompt: {'ON' if self.colorized_prompt else 'OFF'}")
            elif ch in (ord("r"), ord("R")):
                scans = self.sys.scan_once()
                with self._lock:
                    self.scans = scans
                self.log("Forced rescan.")
            elif ch in (ord("a"), ord("A")):
                self._run_ai_for_focus()

    def _draw(self, stdscr) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        dash_h = 9
        log_h = 6
        sig_h = 10
        mid_h = h - dash_h - log_h
        sig_h = min(sig_h, max(6, mid_h // 2))

        focus = self.sys.get_focus_domain()
        stdscr.addstr(
            0,
            2,
            f"RGN CEB META-PROMPT SYSTEM | Focus: {focus} | Q quit TAB domain P prompt O output C color R rescan A AI",
            curses.A_BOLD,
        )

        with self._lock:
            scans_copy = dict(self.scans)
            logs_copy = list(self.logs)

        self._draw_dashboard(stdscr, scans_copy, y=1, x=2, height=dash_h - 1, width=w - 4, focus=focus)
        self._draw_signature(stdscr, scans_copy, y=dash_h, x=2, height=sig_h, width=max(30, w // 3), focus=focus)

        if self.show_prompt:
            self._draw_prompt_panel(
                stdscr,
                scans_copy,
                y=dash_h,
                x=2 + max(30, w // 3) + 1,
                height=mid_h,
                width=w - (2 + max(30, w // 3) + 3),
                focus=focus,
            )

        self._draw_logs(stdscr, logs_copy, y=h - log_h, x=2, height=log_h, width=w - 4)
        stdscr.refresh()

    def _draw_dashboard(self, stdscr, scans: Dict[str, DomainScan], y: int, x: int, height: int, width: int, focus: str) -> None:
        stdscr.addstr(y, x, "DASHBOARD", curses.A_UNDERLINE)
        headers = ["DOMAIN", "DIAL", "BAND", "DRIFT", "CONF", "VOL", "CEB_H"]
        header_line = " | ".join(hh.ljust(14) for hh in headers)
        stdscr.addstr(y + 1, x, header_line[:width - 1], curses.A_DIM)

        row = y + 2
        for d in self.sys.domains:
            s = scans.get(d)
            if not s:
                continue
            line = " | ".join([
                d.ljust(14),
                f"{s.risk:.3f}".ljust(14),
                s.status.ljust(14),
                f"{s.drift:+.3f}".ljust(14),
                f"{s.confidence:.2f}".ljust(14),
                f"{s.volatility:.4f}".ljust(14),
                f"{s.ceb_entropy:.3f}".ljust(14),
            ])

            attr = curses.A_NORMAL
            if d == focus:
                attr |= curses.A_REVERSE
            if s.status == "HIGH":
                attr |= curses.A_BOLD
            elif s.status == "MODERATE":
                attr |= curses.A_DIM

            if row < y + height:
                stdscr.addstr(row, x, line[:width - 1], attr)
            row += 1

    def _draw_signature(self, stdscr, scans: Dict[str, DomainScan], y: int, x: int, height: int, width: int, focus: str) -> None:
        stdscr.addstr(y, x, "CEB SIGNATURE (top colors)", curses.A_UNDERLINE)
        s = scans.get(focus)
        if not s:
            return

        row = y + 1
        max_items = min(8, height - 2)
        for item in s.ceb_top[:max_items]:
            i = int(item["i"])
            p = float(item["p"])
            r, g, b = item["rgb"]

            bar_len = int((width - 18) * min(1.0, p * 8))
            bar_len = max(0, min(width - 18, bar_len))
            bar = "█" * bar_len

            line = f"CEB {i:02d} p={p:.4f} "
            if row < y + height:
                stdscr.addstr(row, x, line[:width - 1], curses.A_DIM)
                color_idx = rgb_to_xterm256(int(r), int(g), int(b))
                pair_id = self._color_cache.get_pair(color_idx)
                attr = curses.color_pair(pair_id) | curses.A_BOLD if pair_id != 0 else curses.A_BOLD
                stdscr.addstr(row, x + len(line), bar[: max(0, width - len(line) - 1)], attr)
            row += 1

        if row < y + height:
            stdscr.addstr(row, x, f"Entropy={s.ceb_entropy:.4f}", curses.A_DIM)

    def _draw_prompt_panel(self, stdscr, scans: Dict[str, DomainScan], y: int, x: int, height: int, width: int, focus: str) -> None:
        title = "META-PROMPT (focused domain)"
        if self.show_ai_output:
            title += " + AI OUTPUT (O toggles)"
        stdscr.addstr(y, x, title, curses.A_UNDERLINE)

        s = scans.get(focus)
        if not s or not s.prompt_plan:
            return

        plan = s.prompt_plan
        meta = f"temp={plan.temperature:.2f} max_tokens={plan.max_tokens} chars={plan.meta.get('chars', 0)}"
        stdscr.addstr(y + 1, x, meta[:width - 1], curses.A_DIM)

        if self.show_ai_output:
            text = s.last_ai_output.strip() or "(no AI output yet; press A)"
        else:
            text = plan.prompt
            if not self.colorized_prompt:
                text = strip_rgb_tags(text)

        lines = text.splitlines()
        row = y + 2
        max_lines = height - 3
        for ln in lines[: max(0, max_lines)]:
            if row >= y + height - 1:
                break
            stdscr.addstr(row, x, ln[:width - 1])
            row += 1

    def _draw_logs(self, stdscr, logs: List[str], y: int, x: int, height: int, width: int) -> None:
        stdscr.addstr(y, x, "LOGS", curses.A_UNDERLINE)
        show = logs[-(height - 1):]
        row = y + 1
        for ln in show:
            if row >= y + height:
                break
            stdscr.addstr(row, x, ln[:width - 1], curses.A_DIM)
            row += 1

    def _run_ai_for_focus(self) -> None:
        focus = self.sys.get_focus_domain()
        with self._lock:
            s = self.scans.get(focus)

        if not s or not s.prompt_plan:
            self.log("No prompt available.")
            return
        if not OPENAI_API_KEY:
            self.log("OPENAI_API_KEY not set.")
            return

        self.log(f"AI run start for {focus}...")

        def worker():
            try:
                client = HttpxOpenAIClient(api_key=OPENAI_API_KEY)
                out = client.chat(
                    prompt=s.prompt_plan.prompt,
                    temperature=s.prompt_plan.temperature,
                    max_tokens=s.prompt_plan.max_tokens,
                    retries=3,
                )
                with self._lock:
                    if focus in self.scans:
                        self.scans[focus].last_ai_output = out
                self.log(f"AI run done for {focus}.")
            except Exception as e:
                self.log(f"AI error: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()


def strip_rgb_tags(prompt: str) -> str:
    out = []
    for ln in prompt.splitlines():
        s = ln.strip()
        if s.startswith("<RGB "):
            continue
        if s.startswith("</RGB"):
            continue
        out.append(ln)
    return "\n".join(out)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    sys = RGNCebSystem(domains=DEFAULT_DOMAINS)
    tui = AdvancedTUI(sys)
    tui.run()


if __name__ == "__main__":
    main()
