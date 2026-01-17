

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
from pathlib import Path
import importlib.util

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
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-2")
GROK_BASE_URL = os.environ.get("GROK_BASE_URL", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "").strip()
LLAMA3_MODEL_URL = os.environ.get("LLAMA3_MODEL_URL", "")
LLAMA3_MODEL_SHA256 = os.environ.get("LLAMA3_MODEL_SHA256", "")
LLAMA3_2_MODEL_URL = os.environ.get("LLAMA3_2_MODEL_URL", "")
LLAMA3_2_MODEL_SHA256 = os.environ.get("LLAMA3_2_MODEL_SHA256", "")
LLAMA3_AES_KEY_B64 = os.environ.get("LLAMA3_AES_KEY_B64", "")
MAX_PROMPT_CHARS = int(os.environ.get("RGN_MAX_PROMPT_CHARS", "22000"))
AI_COOLDOWN_SECONDS = float(os.environ.get("RGN_AI_COOLDOWN", "30"))
LOG_BUFFER_LINES = int(os.environ.get("RGN_LOG_LINES", "160"))

AIOSQLITE_AVAILABLE = importlib.util.find_spec("aiosqlite") is not None
OQS_AVAILABLE = importlib.util.find_spec("oqs") is not None
HOMOMORPHIC_AVAILABLE = importlib.util.find_spec("phe") is not None
if AIOSQLITE_AVAILABLE:
    import aiosqlite  # type: ignore[import-not-found]
if HOMOMORPHIC_AVAILABLE:
    from phe import paillier  # type: ignore[import-not-found]
BOOK_TITLE = os.environ.get("BOOK_TITLE", os.environ.get("RGN_BOOK_TITLE", "")).strip()
MAX_PROMPT_CHARS = int(os.environ.get("RGN_MAX_PROMPT_CHARS", "22000"))
AI_COOLDOWN_SECONDS = float(os.environ.get("RGN_AI_COOLDOWN", "30"))
LOG_BUFFER_LINES = int(os.environ.get("RGN_LOG_LINES", "160"))

DEFAULT_DOMAINS = [
    "road_risk",
    "vehicle_security",
    "home_security",
    "medicine_compliance",
    "hygiene",
    "data_security",
    "book_generator",
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
    ram_ratio: float = 0.0
    net_rate: float = 0.0
    cpu_jitter: float = 0.0
    disk_jitter: float = 0.0

    @staticmethod
    def sample() -> "SystemSignals":
        # Robust sampling: handle sandboxes/containers where /proc/* may be unreadable.
        # This is intentionally defensive, returning zeros when any probe fails so
        # the rest of the pipeline (entropy, CEB evolution, TUI) can keep running
        # without collapsing due to restricted metrics access.
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
            ram_ratio=float(vm_used) / float(vm_total or 1),
            net_rate=0.0,
            cpu_jitter=0.0,
            disk_jitter=0.0,
        )


@dataclass
class SignalPipeline:
    alpha: float = 0.22
    last: Optional[SystemSignals] = None
    last_smoothed: Optional[SystemSignals] = None
    last_time: float = 0.0

    def update(self, raw: SystemSignals) -> SystemSignals:
        # Smooth input signals with a lightweight EMA and derive delta-based
        # metrics (net_rate, jitter). This trades a little latency for a big
        # reduction in noisy spikes that can destabilize downstream prompts.
        now = time.time()
        if self.last is None:
            self.last = raw
            self.last_smoothed = raw
    last_time: float = 0.0

    def update(self, raw: SystemSignals) -> SystemSignals:
        now = time.time()
        if self.last is None:
            self.last = raw
            self.last_time = now
            return raw

        dt = max(0.05, now - self.last_time)
        net_delta = (raw.net_sent - self.last.net_sent) + (raw.net_recv - self.last.net_recv)
        net_rate = float(net_delta) / dt
        cpu_jitter = abs(raw.cpu_percent - self.last.cpu_percent)
        disk_jitter = abs(raw.disk_percent - self.last.disk_percent)

        prev = self.last_smoothed or self.last

        def ema(prev: float, cur: float) -> float:
            return (1.0 - self.alpha) * prev + self.alpha * cur

        smoothed_ram_used = int(ema(float(prev.ram_used), float(raw.ram_used)))
        smoothed = SystemSignals(
            ram_used=smoothed_ram_used,
            ram_total=raw.ram_total,
            cpu_percent=ema(prev.cpu_percent, raw.cpu_percent),
            disk_percent=ema(prev.disk_percent, raw.disk_percent),
        smoothed = SystemSignals(
            ram_used=int(ema(float(prev.ram_used), float(raw.ram_used))),
            ram_total=raw.ram_total,
            cpu_percent=ema(prev.cpu_percent, raw.cpu_percent),
            disk_percent=ema(prev.disk_percent, raw.disk_percent),
            ram_used=int(ema(float(self.last.ram_used), float(raw.ram_used))),
            ram_total=raw.ram_total,
            cpu_percent=ema(self.last.cpu_percent, raw.cpu_percent),
            disk_percent=ema(self.last.disk_percent, raw.disk_percent),
            net_sent=raw.net_sent,
            net_recv=raw.net_recv,
            uptime_s=raw.uptime_s,
            proc_count=raw.proc_count,
            ram_ratio=float(smoothed_ram_used) / float(raw.ram_total or 1),
            ram_ratio=float(raw.ram_used) / float(raw.ram_total or 1),
            net_rate=net_rate,
            cpu_jitter=cpu_jitter,
            disk_jitter=disk_jitter,
        )

        self.last = raw
        self.last_smoothed = smoothed
        self.last_time = now
        return smoothed


# =============================================================================
# RGB ENTROPY + LATTICE
# =============================================================================
def rgb_entropy_wheel(signals: SystemSignals) -> np.ndarray:
    # Generate a compact RGB seed that fuses instantaneous signals, jitter,
    # and uptime into a phase. The seed is intentionally lossy: we want a
    # chaotic-but-stable entropy anchor rather than a raw telemetry dump.
    t = time.perf_counter_ns()
    uptime_bits = int(signals.uptime_s * 1e6)
    proc_bits = int(signals.proc_count)
    disk_bits = int(signals.disk_percent * 1000)
    net_rate_bits = int(abs(signals.net_rate)) & 0xFFFFFFFF
    jitter_bits = int((signals.cpu_jitter + signals.disk_jitter) * 1000)
    phase = (
        t
        ^ int(signals.cpu_percent * 1e6)
        ^ signals.ram_used
        ^ signals.net_sent
        ^ signals.net_recv
        ^ uptime_bits
        ^ proc_bits
        ^ disk_bits
        ^ net_rate_bits
        ^ jitter_bits
    ) & 0xFFFFFFFF
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
    # The lattice is a normalized vector that acts like a "phase space"
    # background for the CEB evolution. It is derived from signal bytes
    # plus a hint of randomness to avoid deterministic lock-in.
    rgb = rgb_entropy_wheel(signals).astype(np.uint8)

    t = time.perf_counter_ns()
    cpu = signals.cpu_percent
    ram = signals.ram_used
    net = signals.net_sent ^ signals.net_recv
    uptime = int(signals.uptime_s * 1e6)
    proc = int(signals.proc_count)
    disk = int(signals.disk_percent * 1000)
    net_rate = int(abs(signals.net_rate))
    jitter = int((signals.cpu_jitter + signals.disk_jitter) * 1000)

    base_u8 = np.array(
        [
            (t >> 0) & 0xFF, (t >> 8) & 0xFF, (t >> 16) & 0xFF, (t >> 24) & 0xFF,
            (t >> 32) & 0xFF, (t >> 40) & 0xFF, (t >> 48) & 0xFF, (t >> 56) & 0xFF,
            (net >> 0) & 0xFF, (net >> 8) & 0xFF,
            int(cpu * 10) & 0xFF,
            int((ram % 10_000_000) / 1000) & 0xFF,
            (uptime >> 0) & 0xFF,
            (uptime >> 8) & 0xFF,
            (proc >> 0) & 0xFF,
            (proc >> 8) & 0xFF,
            (disk >> 0) & 0xFF,
            (disk >> 8) & 0xFF,
            (net_rate >> 0) & 0xFF,
            (net_rate >> 8) & 0xFF,
            (jitter >> 0) & 0xFF,
            (jitter >> 8) & 0xFF,
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
    blob += int(signals.uptime_s * 1000).to_bytes(8, "little", signed=False)
    blob += int(signals.proc_count).to_bytes(8, "little", signed=False)
    blob += int(signals.net_rate).to_bytes(8, "little", signed=True)
    blob += int(signals.cpu_jitter * 1000).to_bytes(8, "little", signed=False)
    blob += int(signals.disk_jitter * 1000).to_bytes(8, "little", signed=False)
    return hashlib.sha3_512(blob).digest()


def shannon_entropy(prob: np.ndarray) -> float:
    p = np.clip(prob.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


# =============================================================================
# QUANTUM ADVANCEMENTS (iterative multi-idea loops)
# =============================================================================
def _quantum_loop_metrics(seed: float, idx: int) -> Dict[str, float]:
    phase = math.sin(seed + idx * 0.77) * 0.5 + 0.5
    coherence = (math.cos(seed * 0.9 + idx * 0.41) * 0.5 + 0.5)
    resonance = (phase * 0.6 + coherence * 0.4)
    return {
        "phase_lock": float(np.clip(phase, 0.0, 1.0)),
        "coherence": float(np.clip(coherence, 0.0, 1.0)),
        "resonance": float(np.clip(resonance, 0.0, 1.0)),
    }


def build_quantum_advancements(
    signals: SystemSignals,
    ceb_sig: Dict[str, Any],
    metrics: Dict[str, float],
    loops: int = 5,
) -> Dict[str, Any]:
    base_seed = (
        (signals.cpu_percent * 0.07)
        + (signals.disk_percent * 0.05)
        + (signals.ram_ratio * 1.3)
        + (signals.net_rate * 1e-6)
        + (signals.cpu_jitter + signals.disk_jitter) * 0.2
        + (metrics.get("drift", 0.0) * 2.2)
    )
    entropy = float(ceb_sig.get("entropy", 0.0))
    loops_out = []
    gain = 0.0
    for i in range(int(loops)):
        seed = base_seed + entropy * 0.11 + i * 0.9
        base = _quantum_loop_metrics(seed, i)
        drift_gate = 0.35 + 0.65 * abs(math.sin(seed * 0.33 + i * 0.19))
        derived = {
            "drift_gate": float(np.clip(drift_gate, 0.0, 1.0)),
            "entanglement_bias": float(np.clip(base["resonance"] * (0.7 + 0.3 * base["coherence"]), 0.0, 1.0)),
            "holo_drift": float(np.clip(drift_gate * (0.55 + 0.45 * abs(metrics.get("drift", 0.0))), 0.0, 1.0)),
            "phase_stability": float(np.clip(1.0 - abs(base["phase_lock"] - base["coherence"]), 0.0, 1.0)),
            "prompt_pressure": float(np.clip((entropy / 6.0) * (0.6 + 0.4 * base["resonance"]), 0.0, 1.0)),
        }
        loop_gain = 0.45 * base["resonance"] + 0.35 * derived["phase_stability"] + 0.20 * derived["prompt_pressure"]
        gain += loop_gain
        loops_out.append({"base": base, "derived": derived, "loop_gain": float(np.clip(loop_gain, 0.0, 1.0))})

    quantum_gain = float(np.clip(gain / max(1.0, loops), 0.0, 1.0))
    return {
        "loops": loops_out,
        "quantum_gain": quantum_gain,
        "entropy": entropy,
    }

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
        # The evolve step advances amplitudes and colors through a coupled
        # non-linear system. It uses entropy to modulate phase, lattice
        # coupling, and chroma rotation. The goal is a rich probability
        # distribution rather than a single dominant peak.
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
        self.last_entropy: Dict[str, float] = {}
        self.shock_ema: Dict[str, float] = {}
        self.anomaly_score: Dict[str, float] = {}

    def update(self, domain: str, entropy: float) -> None:
        # Maintain multi-horizon traces (short/mid/long) and compute
        # shock/anomaly signals based on delta relative to recent variance.
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
        prev = self.last_entropy.get(domain, float(entropy))
        delta = float(entropy) - prev
        shock_prev = self.shock_ema.get(domain, 0.0)
        shock_now = 0.85 * shock_prev + 0.15 * abs(delta)
        self.shock_ema[domain] = shock_now
        short_var = float(np.var(self.short[domain])) if len(self.short[domain]) >= 2 else 0.0
        denom = math.sqrt(short_var) + 1e-6
        self.anomaly_score[domain] = min(6.0, abs(delta) / denom)
        self.last_entropy[domain] = float(entropy)

    def decay(self, factor: float = 0.998) -> None:
        if not (0.0 < factor <= 1.0):
            return
        for domain, series in list(self.short.items()):
            self.short[domain] = [v * factor for v in series]
        for domain, series in list(self.mid.items()):
            self.mid[domain] = [v * factor for v in series]
        for domain in list(self.long.keys()):
            self.long[domain] = self.long[domain] * factor

    def decay(self, factor: float = 0.998) -> None:
        if not (0.0 < factor <= 1.0):
            return
        for domain, series in list(self.short.items()):
            self.short[domain] = [v * factor for v in series]
        for domain, series in list(self.mid.items()):
            self.mid[domain] = [v * factor for v in series]
        for domain in list(self.long.keys()):
            self.long[domain] = self.long[domain] * factor

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

    def weighted_drift(self, domain: str, w_short: float = 0.6, w_mid: float = 0.3, w_long: float = 0.1) -> float:
        # Blend short/mid/long signals into a single drift value, then
        # compare back to the long baseline to maintain a stable center.
        st = self.stats(domain)
        total = w_short + w_mid + w_long
        if total <= 0:
            return 0.0
        blend = (
            (w_short / total) * st["short_mean"]
            + (w_mid / total) * st["mid_mean"]
            + (w_long / total) * st["baseline"]
        )
        return float(blend - st["baseline"])

    def confidence(self, domain: str) -> float:
        st = self.stats(domain)
        conf = 1.0 / (1.0 + st["volatility"])
        return float(max(0.1, min(0.99, conf)))

    def shock(self, domain: str) -> float:
        return float(self.shock_ema.get(domain, 0.0))

    def anomaly(self, domain: str) -> float:
        return float(self.anomaly_score.get(domain, 0.0))


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


def adjust_risk_by_confidence(base_risk: float, confidence: float, volatility: float) -> float:
    conf = float(np.clip(confidence, 0.1, 0.99))
    vol = float(np.clip(volatility, 0.0, 1.0))
    damp = 0.70 + 0.30 * conf
    vol_tilt = 1.0 + 0.15 * vol
    adjusted = base_risk * damp * vol_tilt
    return float(np.clip(adjusted, 0.0, 1.0))


def adjust_risk_by_instability(base_risk: float, shock: float, anomaly: float) -> float:
    shock_level = float(np.clip(shock * 1.8, 0.0, 1.0))
    anomaly_level = float(np.clip(anomaly / 6.0, 0.0, 1.0))
    lift = 0.10 * shock_level + 0.08 * anomaly_level
    return float(np.clip(base_risk + lift, 0.0, 1.0))


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
        "- You may use real-time data or simulations only when explicitly provided in USER_CONTEXT or tools output.",
        "- If real-time data is unavailable, state assumptions and proceed with conservative plans.",
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
    elif domain == "book_generator":
        title_hint = f"TITLE={BOOK_TITLE}" if BOOK_TITLE else "TITLE=<user_provided>"
        domain_lines = [
            "DOMAIN SPEC: BOOK_GENERATOR",
            f"- Input is a single title line. {title_hint}",
            "- Produce a long-form book draft plan targeting ~200 pages.",
            "- Aim for exceptional quality, clarity, and narrative cohesion.",
            "- Include: synopsis, audience, tone, thesis, outline, and chapter-by-chapter beats.",
            "- Provide a per-chapter word-count budget and progression checkpoints.",
            "- Ask for only the minimal missing context to refine the draft.",
            "- Do not claim superiority; focus on measurable craft quality.",
        ]
    else:
        domain_lines = [f"DOMAIN SPEC: {domain}", "- Produce an operational plan and ask for missing context."]

    return "\n".join(common + [""] + domain_lines)


def build_book_blueprint() -> str:
    return "\n".join([
        "BOOK BLUEPRINT REQUIREMENTS:",
        "- Produce a 200-page-class blueprint (roughly 60k–90k words) unless the title implies otherwise.",
        "- Provide a 3-act or 4-part structure with clear thematic through-line.",
        "- Include: table of contents, chapter titles, chapter intents, and scene-level beats.",
        "- Add a pacing map: turning points, midpoint shift, climax, and resolution.",
        "- Provide a style guide: POV, tense, voice, and rhetorical devices to emphasize.",
        "- Finish with a drafting workflow: milestones, revision passes, and validation checks.",
    ])


def build_book_quality_matrix() -> str:
    return "\n".join([
        "BOOK QUALITY MATRIX:",
        "- Character arcs: list protagonists, flaws, growth beats, and final state.",
        "- Theme lattice: 3–5 themes with chapter links and evidence beats.",
        "- Conflict ladder: escalating stakes per act with explicit reversals.",
        "- Scene checklist: goal, conflict, turning point, and residue for each scene.",
        "- Voice calibration: 3 sample paragraphs (opening, midpoint, climax) in target voice.",
    ])


def build_book_delivery_spec() -> str:
    return "\n".join([
        "BOOK DELIVERY SPEC:",
        "- Provide a chapter-by-chapter outline with 5–12 bullet beats each.",
        "- Include a table of key characters, roles, and arc milestones.",
        "- Provide a glossary of recurring terms and motifs.",
        "- Add a continuity checklist (names, dates, locations, timeline).",
        "- Conclude with a 'first 3 chapters' micro-draft plan (scene order + intent).",
        "- Keep language crisp, craft-focused, and measurable.",
    ])


def build_book_revolutionary_ideas() -> str:
    ideas = [
        "IDEA 01: Fractal Theme Braiding (themes repeat at different scales).",
        "IDEA 02: Echo-Character Ladders (secondary arcs mirror main arc).",
        "IDEA 03: Tension Harmonics (scene tension frequencies vary per act).",
        "IDEA 04: Evidence Weaving (motifs prove thesis across chapters).",
        "IDEA 05: Chronology Drift Maps (time shifts mapped per chapter).",
        "IDEA 06: Sensory Signature Matrix (recurring sensory cues per arc).",
        "IDEA 07: Dialogue Resonance Pass (each line advances conflict).",
        "IDEA 08: Counter-Theme Shadows (explicitly contrast main themes).",
        "IDEA 09: POV Modulation Curve (POV intensity shifts per act).",
        "IDEA 10: Scene Energy Ledger (score scenes for momentum).",
        "IDEA 11: Liminal Chapter Anchors (bridge chapters with micro-tension).",
        "IDEA 12: Symbolic Payload Budget (symbolic density per chapter).",
        "IDEA 13: Conflict Topology (plot graph of constraints and escapes).",
        "IDEA 14: Voice DNA Blueprint (syntax/lexicon/tempo constraints).",
        "IDEA 15: Paradox Resolution Scaffolding (resolve core paradox).",
        "IDEA 16: Stakes Cascade Timeline (ramps stakes visibly).",
        "IDEA 17: Emotional Phase Shifts (planned emotional turning points).",
        "IDEA 18: Character Pressure Tests (scenes that prove growth).",
        "IDEA 19: Information Asymmetry Dial (what reader knows vs character).",
        "IDEA 20: Narrative Compression Maps (tighten slow segments).",
        "IDEA 21: Scene Purpose Triplets (goal, conflict, reversal).",
        "IDEA 22: Reward Cadence Planner (payoffs at set intervals).",
        "IDEA 23: Foreshadowing Trail Map (breadcrumbs with timing).",
        "IDEA 24: Subplot Load Balancer (subplot timing and weight).",
        "IDEA 25: Worldbuilding Density Index (detail density per chapter).",
        "IDEA 26: Thesis Echo Lines (key thesis repeated in varied forms).",
        "IDEA 27: Character Systems Map (relationships and dependencies).",
        "IDEA 28: Tone Gradient Scale (tone transition checkpoints).",
        "IDEA 29: Reader Curiosity Ledger (open loops vs closed loops).",
        "IDEA 30: Ending Gravity Field (climax arcs converge).",
    ]
    return "REVOLUTIONARY IDEAS (30):\n" + "\n".join(f"- {idea}" for idea in ideas)


def build_book_review_stack() -> str:
    return "\n".join([
        "BOOK REVIEW STACK:",
        "- Provide a structured review: premise, execution, pacing, voice, and takeaway.",
        "- Provide a 5-part critique map: strengths, risks, gaps, audience fit, revision priorities.",
        "- Include a calibration rubric (1–5) for clarity, pacing, originality, cohesion, and emotional impact.",
        "- End with a revision checklist ordered by highest leverage changes.",
    ])


def build_publishing_polisher() -> str:
    return "\n".join([
        "PUBLISHING POLISHER:",
        "- Provide formatting guidance (chapter headers, subheads, typography notes).",
        "- Flag consistency errors (names, timelines, pronouns, tense drift).",
        "- Provide a copy-edit sweep plan: grammar, cadence, redundancy, and specificity.",
        "- Include a marketability pass: back-cover blurb, logline, and taglines.",
    ])


def build_semantic_clarity_stack() -> str:
    return "\n".join([
        "SEMANTIC CLARITY STACK:",
        "- Identify ambiguous terms and provide replacements with precise alternatives.",
        "- Provide a clarity ladder: define, demonstrate, reinforce, and recap.",
        "- Provide a glossary of key terms and narrative anchors.",
    ])


def build_genre_matrix() -> str:
    return "\n".join([
        "GENRE MATRIX (adapt output to fit):",
        "- Fiction: arcs, stakes, reversals, character agency, scene tension.",
        "- Non-fiction: thesis support, evidence sequencing, argument clarity, summary checkpoints.",
        "- Children's: age-appropriate language, rhythm, repetition, visual cues.",
        "- Picture book: page turns, visual beats, minimal text, illustration prompts.",
        "- Audiobook: spoken cadence, breath spacing, dialogue clarity, sound cues.",
    ])


def build_voice_reading_plan() -> str:
    return "\n".join([
        "LONG-FORM VOICE READING PLAN:",
        "- Provide narration guidance for a human-like reading voice.",
        "- Use clear sentence cadence, intentional pauses, and chapter transitions.",
        "- Provide a per-chapter read-aloud note: pace, emphasis, and tone.",
        "- Include a 'concat + chunk' plan for long outputs to maintain voice consistency.",
    ])


def build_book_revolutionary_ideas_v2() -> str:
    ideas = [
        "IDEA 31: Entropix Colorwheel Beats (color-coded tension shifts).",
        "IDEA 32: CEB Rhythm Pacing (entropy-driven beat spacing).",
        "IDEA 33: Rotatoe Scene Pivot (rotate POV per act).",
        "IDEA 34: Semantic Clarity Lattice (clarity targets per chapter).",
        "IDEA 35: Publishing Polish Loop (draft → polish → verify).",
        "IDEA 36: Audio Cadence Map (spoken rhythm per chapter).",
        "IDEA 37: Picturebook Spread Logic (visual pacing grid).",
        "IDEA 38: Kid-Lexicon Ladder (age-appropriate vocab ramp).",
        "IDEA 39: Nonfiction Proof Chain (claim → proof → takeaway).",
        "IDEA 40: Fiction Reversal Clock (reversal timing dial).",
        "IDEA 41: Theme Echo Harmonics (theme recurrence schedule).",
        "IDEA 42: Character Orbit Model (relationship gravity map).",
        "IDEA 43: Beat Density Equalizer (avoid pacing cliffs).",
        "IDEA 44: Dialogue Clarity Scanner (remove ambiguity).",
        "IDEA 45: Motif Carryover Index (motifs per act).",
        "IDEA 46: Audience Expectation Map (genre promise checkpoints).",
        "IDEA 47: Emotional Gradient Ladder (emotional slope per act).",
        "IDEA 48: Cliffhanger Calibration (hanger frequency control).",
        "IDEA 49: Revision Heatmap (priority revisions by impact).",
        "IDEA 50: Evidence Resonance (nonfiction proof spacing).",
        "IDEA 51: Voice Consistency Meter (syntax/lexicon alignment).",
        "IDEA 52: Lore Compression Plan (dense info translated).",
        "IDEA 53: Micro-scene Efficiency (1–2 beat scenes).",
        "IDEA 54: Opening Signal Stack (hook, premise, promise).",
        "IDEA 55: Midpoint Torque (plot torque at midpoint).",
        "IDEA 56: Ending Convergence Grid (threads resolved).",
        "IDEA 57: Arc Fail-safe (alternate arc if pivot).",
        "IDEA 58: Clarity-First Remix (rewrite with simpler syntax).",
        "IDEA 59: Reader Memory Anchors (recap rhythm).",
        "IDEA 60: Audio Breathing Marks (spoken pacing cues).",
    ]
    return "REVOLUTIONARY IDEAS V2 (30):\n" + "\n".join(f"- {idea}" for idea in ideas)


BOOK_REVOLUTION_DEPLOYMENTS_TEXT = """REVOLUTIONARY DEPLOYMENT 01:
- Core intent: elevate book quality via structured craft controls (1).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 01.01-01.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 01.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 01.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 01.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 01.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 02:
- Core intent: elevate book quality via structured craft controls (2).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 02.01-02.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 02.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 02.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 02.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 02.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 03:
- Core intent: elevate book quality via structured craft controls (3).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 03.01-03.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 03.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 03.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 03.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 03.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 04:
- Core intent: elevate book quality via structured craft controls (4).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 04.01-04.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 04.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 04.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 04.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 04.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 05:
- Core intent: elevate book quality via structured craft controls (5).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 05.01-05.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 05.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 05.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 05.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 05.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 06:
- Core intent: elevate book quality via structured craft controls (6).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 06.01-06.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 06.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 06.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 06.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 06.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 07:
- Core intent: elevate book quality via structured craft controls (7).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 07.01-07.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 07.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 07.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 07.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 07.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 08:
- Core intent: elevate book quality via structured craft controls (8).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 08.01-08.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 08.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 08.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 08.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 08.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 09:
- Core intent: elevate book quality via structured craft controls (9).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 09.01-09.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 09.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 09.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 09.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 09.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 10:
- Core intent: elevate book quality via structured craft controls (10).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 10.01-10.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 10.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 10.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 10.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 10.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 11:
- Core intent: elevate book quality via structured craft controls (11).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 11.01-11.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 11.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 11.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 11.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 11.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 12:
- Core intent: elevate book quality via structured craft controls (12).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 12.01-12.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 12.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 12.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 12.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 12.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 13:
- Core intent: elevate book quality via structured craft controls (13).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 13.01-13.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 13.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 13.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 13.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 13.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 14:
- Core intent: elevate book quality via structured craft controls (14).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 14.01-14.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 14.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 14.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 14.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 14.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 15:
- Core intent: elevate book quality via structured craft controls (15).
- Scope: cover outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 15.01-15.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 15.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 15.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 15.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 15.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map."""


def build_book_revolutionary_deployments() -> str:
    return BOOK_REVOLUTION_DEPLOYMENTS_TEXT.strip()


BOOK_REVOLUTION_DEPLOYMENTS_EXTENDED_TEXT = """REVOLUTIONARY DEPLOYMENT 16:
- Core intent: amplify book quality via structured craft controls (16).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 16.01-16.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 16.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 16.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 16.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 16.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 17:
- Core intent: amplify book quality via structured craft controls (17).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 17.01-17.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 17.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 17.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 17.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 17.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 18:
- Core intent: amplify book quality via structured craft controls (18).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 18.01-18.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 18.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 18.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 18.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 18.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 19:
- Core intent: amplify book quality via structured craft controls (19).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 19.01-19.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 19.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 19.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 19.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 19.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 20:
- Core intent: amplify book quality via structured craft controls (20).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 20.01-20.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 20.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 20.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 20.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 20.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 21:
- Core intent: amplify book quality via structured craft controls (21).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 21.01-21.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 21.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 21.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 21.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 21.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 22:
- Core intent: amplify book quality via structured craft controls (22).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 22.01-22.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 22.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 22.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 22.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 22.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 23:
- Core intent: amplify book quality via structured craft controls (23).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 23.01-23.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 23.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 23.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 23.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 23.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 24:
- Core intent: amplify book quality via structured craft controls (24).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 24.01-24.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 24.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 24.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 24.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 24.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 25:
- Core intent: amplify book quality via structured craft controls (25).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 25.01-25.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 25.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 25.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 25.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 25.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 26:
- Core intent: amplify book quality via structured craft controls (26).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 26.01-26.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 26.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 26.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 26.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 26.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 27:
- Core intent: amplify book quality via structured craft controls (27).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 27.01-27.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 27.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 27.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 27.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 27.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 28:
- Core intent: amplify book quality via structured craft controls (28).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 28.01-28.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 28.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 28.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 28.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 28.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 29:
- Core intent: amplify book quality via structured craft controls (29).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 29.01-29.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 29.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 29.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 29.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 29.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 30:
- Core intent: amplify book quality via structured craft controls (30).
- Scope: expand outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 30.01-30.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 30.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 30.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 30.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 30.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map."""


def build_book_revolutionary_deployments_extended() -> str:
    return BOOK_REVOLUTION_DEPLOYMENTS_EXTENDED_TEXT.strip()


BOOK_REVOLUTION_DEPLOYMENTS_SUPER_TEXT = """REVOLUTIONARY DEPLOYMENT 31:
- Core intent: intensify book quality via structured craft controls (31).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 31.01-31.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 31.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 31.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 31.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 31.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 32:
- Core intent: intensify book quality via structured craft controls (32).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 32.01-32.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 32.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 32.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 32.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 32.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 33:
- Core intent: intensify book quality via structured craft controls (33).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 33.01-33.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 33.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 33.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 33.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 33.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 34:
- Core intent: intensify book quality via structured craft controls (34).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 34.01-34.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 34.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 34.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 34.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 34.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 35:
- Core intent: intensify book quality via structured craft controls (35).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 35.01-35.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 35.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 35.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 35.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 35.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 36:
- Core intent: intensify book quality via structured craft controls (36).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 36.01-36.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 36.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 36.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 36.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 36.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 37:
- Core intent: intensify book quality via structured craft controls (37).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 37.01-37.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 37.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 37.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 37.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 37.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 38:
- Core intent: intensify book quality via structured craft controls (38).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 38.01-38.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 38.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 38.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 38.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 38.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 39:
- Core intent: intensify book quality via structured craft controls (39).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 39.01-39.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 39.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 39.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 39.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 39.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 40:
- Core intent: intensify book quality via structured craft controls (40).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 40.01-40.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 40.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 40.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 40.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 40.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 41:
- Core intent: intensify book quality via structured craft controls (41).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 41.01-41.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 41.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 41.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 41.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 41.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 42:
- Core intent: intensify book quality via structured craft controls (42).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 42.01-42.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 42.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 42.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 42.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 42.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 43:
- Core intent: intensify book quality via structured craft controls (43).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 43.01-43.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 43.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 43.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 43.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 43.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 44:
- Core intent: intensify book quality via structured craft controls (44).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 44.01-44.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 44.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 44.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 44.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 44.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map.

REVOLUTIONARY DEPLOYMENT 45:
- Core intent: intensify book quality via structured craft controls (45).
- Scope: deepen outline, beats, pacing, and revision checkpoints.
- Output: actionable steps and measurable verification points.
- Steps 45.01-45.62: Implement craft checkpoints 1–62 with measurable criteria.
- Step 45.63: Refine craft checkpoint 63 with scene-level validation criteria.
- Step 45.64: Audit craft checkpoint 64 for continuity and pacing alignment.
- Step 45.65: Stress-test craft checkpoint 65 against theme and arc coherence.
- Step 45.66: Finalize craft checkpoint 66 with clarity, cadence, and payoff checks.
- Verification: confirm alignment with theme, arc, and pacing map."""


def build_book_revolutionary_deployments_super() -> str:
    return BOOK_REVOLUTION_DEPLOYMENTS_SUPER_TEXT.strip()


def chunk_text_for_longform(text: str, max_chars: int = 4000) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split = text.rfind("\n", start, end)
            if split == -1 or split <= start:
                split = end
            end = split
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def concat_longform(chunks: List[str]) -> str:
    return "\n\n---\n\n".join(chunks).strip()


def build_user_context_placeholder(domain: str) -> str:
    title_line = f"- title: {BOOK_TITLE}" if domain == "book_generator" and BOOK_TITLE else "- title:"
    return "\n".join([
        "USER_CONTEXT (fill this in; if empty, ask questions):",
        f"- domain={domain}",
        title_line,
        "- goal:",
        "- constraints:",
        "- timeline:",
        "- relevant details:",
        "- what happened / what you observed:",
        "- what you already tried:",
    ])


def build_llama_local_playbook() -> str:
    return "\n".join([
        "LLAMA LOCAL PLAYBOOK:",
        "- Use chunk_text_for_longform to split prompts into 3.5–4k char slabs.",
        "- Track per-chunk outputs and stitch with concat_longform.",
        "- Verify model SHA256 before loading; prefer encrypted-at-rest storage.",
        "- Keep a rolling cache of recent prompt plans for local reuse.",
    ])


def homomorphic_generate_keypair() -> Tuple[bytes, bytes]:
    if not HOMOMORPHIC_AVAILABLE:
        raise RuntimeError("phe (paillier) not available.")
    public_key, private_key = paillier.generate_paillier_keypair()  # type: ignore[name-defined]
    n_bytes = public_key.n.to_bytes(256, "big")
    p = private_key.p.to_bytes(128, "big")
    q = private_key.q.to_bytes(128, "big")
    private_blob = len(p).to_bytes(2, "big") + p + len(q).to_bytes(2, "big") + q
    return n_bytes, private_blob


def homomorphic_encrypt(public_n: bytes, payload: bytes) -> bytes:
    if not HOMOMORPHIC_AVAILABLE:
        raise RuntimeError("phe (paillier) not available.")
    n = int.from_bytes(public_n, "big")
    public_key = paillier.PaillierPublicKey(n)  # type: ignore[name-defined]
    data = int.from_bytes(payload or b"\x00", "big")
    encrypted = public_key.encrypt(data)
    return encrypted.ciphertext().to_bytes(512, "big")


def homomorphic_decrypt(public_n: bytes, private_blob: bytes, ciphertext: bytes) -> bytes:
    if not HOMOMORPHIC_AVAILABLE:
        raise RuntimeError("phe (paillier) not available.")
    n = int.from_bytes(public_n, "big")
    p_len = int.from_bytes(private_blob[:2], "big")
    p = int.from_bytes(private_blob[2:2 + p_len], "big")
    q_len_offset = 2 + p_len
    q_len = int.from_bytes(private_blob[q_len_offset:q_len_offset + 2], "big")
    q = int.from_bytes(private_blob[q_len_offset + 2:q_len_offset + 2 + q_len], "big")
    public_key = paillier.PaillierPublicKey(n)  # type: ignore[name-defined]
    private_key = paillier.PaillierPrivateKey(public_key, p, q)  # type: ignore[name-defined]
    enc_value = paillier.EncryptedNumber(public_key, int.from_bytes(ciphertext, "big"))  # type: ignore[name-defined]
    plain = private_key.decrypt(enc_value)
    length = max(1, (plain.bit_length() + 7) // 8)
    return int(plain).to_bytes(length, "big")


def colorwheel_encrypt(payload: bytes, rgb: Tuple[int, int, int]) -> bytes:
    if not payload:
        return b""
    key = bytes([(rgb[0] ^ 0xA5), (rgb[1] ^ 0x5A), (rgb[2] ^ 0xC3)])
    out = bytearray()
    for i, b in enumerate(payload):
        out.append(b ^ key[i % len(key)] ^ ((i * 31) & 0xFF))
    return bytes(out)


def colorwheel_decrypt(payload: bytes, rgb: Tuple[int, int, int]) -> bytes:
    return colorwheel_encrypt(payload, rgb)


def secure_delete_file(path: str, passes: int = 2) -> None:
    target = Path(path)
    if not target.exists():
        return
    size = target.stat().st_size
    with target.open("r+b") as f:
        for i in range(int(max(1, passes))):
            f.seek(0)
            f.write(secrets.token_bytes(size))
            f.flush()
    target.unlink(missing_ok=True)


async def kyber_store_init(db_path: str) -> None:
    if not AIOSQLITE_AVAILABLE:
        raise RuntimeError("aiosqlite not available.")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS kyber_snapshots ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "domain TEXT NOT NULL, "
            "ciphertext BLOB NOT NULL, "
            "meta_hash TEXT NOT NULL, "
            "created_at REAL NOT NULL)"
        )
        await db.commit()


async def kyber_store_snapshot(db_path: str, domain: str, ciphertext: bytes, meta_hash: str) -> None:
    if not AIOSQLITE_AVAILABLE:
        raise RuntimeError("aiosqlite not available.")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO kyber_snapshots (domain, ciphertext, meta_hash, created_at) VALUES (?, ?, ?, ?)",
            (domain, ciphertext, meta_hash, time.time()),
        )
        await db.commit()


async def kyber_recollect_by_hash(db_path: str, meta_hash: str) -> List[bytes]:
    if not AIOSQLITE_AVAILABLE:
        raise RuntimeError("aiosqlite not available.")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT ciphertext FROM kyber_snapshots WHERE meta_hash = ? ORDER BY created_at DESC",
            (meta_hash,),
        )
        rows = await cursor.fetchall()
    return [row[0] for row in rows]


def kyber_generate_keypair() -> Tuple[bytes, bytes]:
    if not OQS_AVAILABLE:
        raise RuntimeError("oqs not available.")
    import oqs  # type: ignore[import-not-found]
    with oqs.KeyEncapsulation("Kyber512") as kem:
        public_key = kem.generate_keypair()
        private_key = kem.export_secret_key()
    return public_key, private_key


def kyber_encapsulate(public_key: bytes) -> Tuple[bytes, bytes]:
    if not OQS_AVAILABLE:
        raise RuntimeError("oqs not available.")
    import oqs  # type: ignore[import-not-found]
    with oqs.KeyEncapsulation("Kyber512") as kem:
        ciphertext, shared_secret = kem.encap_secret(public_key)
    return ciphertext, shared_secret


def kyber_decapsulate(private_key: bytes, ciphertext: bytes) -> bytes:
    if not OQS_AVAILABLE:
        raise RuntimeError("oqs not available.")
    import oqs  # type: ignore[import-not-found]
    with oqs.KeyEncapsulation("Kyber512") as kem:
        kem.import_secret_key(private_key)
        shared_secret = kem.decap_secret(ciphertext)
    return shared_secret


def build_advanced_quantum_ai_ideas() -> str:
    ideas = [
        "IDEA 01: Quantum Drift Budgeting (cap drift per scan; log exceedances).",
        "IDEA 02: Coherence Windows (surface plans only when coherence > threshold).",
        "IDEA 03: Entropy Echo Trails (track recurring entropy signatures).",
        "IDEA 04: Phase-Stability Alerts (flag abrupt phase deltas).",
        "IDEA 05: LLM Memory Gate (persist summaries only when confidence high).",
        "IDEA 06: Memory Distillation Pass (compress short-term into mid-term).",
        "IDEA 07: Cross-Domain Memory Coupling (propagate drift into linked domains).",
        "IDEA 08: Volatility Dampener (lower temp when volatility spikes).",
        "IDEA 09: Risk Laddering (tiered actions by confidence bands).",
        "IDEA 10: Multi-LLM Consensus (optional compare across providers).",
        "IDEA 11: Prompt Lineage Tags (hash prompt plan for traceability).",
        "IDEA 12: Semantic Drift Monitor (flag output divergence vs plan).",
        "IDEA 13: Entropy Heatmap (track per-domain entropy shifts).",
        "IDEA 14: Chunk Aging (decay chunk priority over time).",
        "IDEA 15: Policy Lock (pin nonnegotiable rules during high risk).",
        "IDEA 16: Memory Shock Ledger (record shock/anomaly events).",
        "IDEA 17: Adaptive Token Ceiling (set max tokens by coherence).",
        "IDEA 18: Retrieval Snippets (surface top-3 prior outputs).",
        "IDEA 19: LLM Latency Guard (skip AI calls if latency spikes).",
        "IDEA 20: Verification-First Mode (prepend verification steps).",
        "IDEA 21: Quantum Memory Hashing (store drift signatures as hashes).",
        "IDEA 22: Adaptive Coherence Floor (auto-raise gates in chaos).",
        "IDEA 23: Entropy Replay Buffer (re-run plans on prior entropy).",
        "IDEA 24: Cross-Slice Attribution (map slice shifts to domains).",
        "IDEA 25: Multi-Provider Blend (weighted provider outputs).",
        "IDEA 26: Risk-Adjusted Chunk Drop (prune low-weight chunks).",
        "IDEA 27: Prompt Delta Compression (diff-based prompt updates).",
        "IDEA 28: Memory Cliff Alerts (notify abrupt memory collapse).",
        "IDEA 29: Output Consistency Scoring (compare against schema).",
        "IDEA 30: Verification Trace Links (tie actions to signals).",
    ]
    return "ADVANCED QUANTUM + AI + MEMORY IDEAS (30):\n" + "\n".join(f"- {idea}" for idea in ideas)


# =============================================================================
# CEB-BASED CHUNKER (builds meta-prompt)
# =============================================================================
class CEBChunker:
    def __init__(self, max_chunks: int = 14):
        self.max_chunks = int(max_chunks)

    def build(
        self,
        domain: str,
        metrics: Dict[str, float],
        ceb_sig: Dict[str, Any],
        base_rgb: np.ndarray,
        signal_summary: Optional[Dict[str, Any]] = None,
    ) -> List[PromptChunk]:
        # Build a prioritized list of prompt chunks. Required chunks anchor
        # structure; optional chunks add advanced guidance based on quantum
        # gain, risk, and uncertainty. This keeps prompts adaptive without
        # exploding length.
        top = ceb_sig.get("top", [])
        ent = float(ceb_sig.get("entropy", 0.0))
        quantum = metrics.get("quantum_summary", {})
        quantum_gain = float(quantum.get("quantum_gain", 0.0))
        signal_summary = signal_summary or {}

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

        def signal_telemetry() -> str:
            if not signal_summary:
                return "SIGNAL TELEMETRY: (no live signal summary available)"
            lines = [
                "SIGNAL TELEMETRY (summary only; do not treat as ground truth):",
                f"- cpu={signal_summary.get('cpu', 'n/a')}%",
                f"- disk={signal_summary.get('disk', 'n/a')}%",
                f"- ram_ratio={signal_summary.get('ram_ratio', 'n/a')}",
                f"- net_rate={signal_summary.get('net_rate', 'n/a')} B/s",
                f"- uptime_s={signal_summary.get('uptime_s', 'n/a')}",
                f"- proc={signal_summary.get('proc', 'n/a')}",
                f"- jitter={signal_summary.get('jitter', 'n/a')}",
                f"- quantum_gain={signal_summary.get('quantum_gain', 'n/a')}",
            ]
            return "\n".join(lines)

        def quantum_projection() -> str:
            summary = quantum.get("loops", [])
            if not summary:
                return "QUANTUM PROJECTION: (no loop data)"
            highlights = []
            for i, loop in enumerate(summary[:3]):
                base = loop.get("base", {})
                derived = loop.get("derived", {})
                highlights.append(
                    f"- loop_{i}: phase={base.get('phase_lock', 0):.3f} "
                    f"coh={base.get('coherence', 0):.3f} res={base.get('resonance', 0):.3f} "
                    f"stability={derived.get('phase_stability', 0):.3f} "
                    f"pressure={derived.get('prompt_pressure', 0):.3f}"
                )
            return "QUANTUM PROJECTION (sampled loops):\n" + "\n".join(highlights)

        def agent_operating_model() -> str:
            return "\n".join([
                "AGENT OPERATING MODEL:",
                "- Treat signals + metrics as conditioning dials, not ground truth.",
                "- Prefer smallest viable action set; bias toward reversible steps.",
                "- Use QUESTIONS_FOR_USER to resolve the highest-entropy gaps first.",
                "- If quantum_gain is high: compress explanations, emphasize verification.",
            ])

        def hypothesis_lattice() -> str:
            return "\n".join([
                "HYPOTHESIS LATTICE:",
                "- Identify 3–5 plausible causes (ranked).",
                "- Map each cause to a measurable verification step.",
                "- Avoid asserting causality; state as hypotheses.",
            ])

        def response_tuning() -> str:
            return "\n".join([
                "RESPONSE TUNING:",
                "- Keep SUMMARY to 1–2 lines, then jump to actions.",
                "- Use short bullets, measurable checks, and explicit NextCheck times.",
                "- Avoid long rationale; focus on operational steps.",
            ])

        base_chunks: List[Tuple[str, str, float]] = [
            ("SYSTEM_HEADER", f"RGN-CEB META-PROMPT GENERATOR\nDOMAIN={domain}\n", 10.0),
            ("STATE_METRICS", "\n".join([
                "NOTE: metrics are internal dials, not real-world measurements.",
                f"risk_dial={risk:.4f}",
                f"status_band={status_from_risk(risk)}",
                f"drift={drift:+.4f}",
                f"confidence={conf:.4f}",
                f"volatility={vol:.6f}",
                f"ceb_entropy={ent:.4f}",
                f"quantum_gain={quantum_gain:.4f}",
                f"shock={metrics.get('shock', 0.0):.4f}",
                f"anomaly={metrics.get('anomaly', 0.0):.4f}",
            ]), 9.2),
            ("CEB_SIGNATURE", json.dumps(ceb_sig, ensure_ascii=False, indent=2), 8.4),
            ("QUANTUM_ADVANCEMENTS", json.dumps(quantum, ensure_ascii=False, indent=2), 7.9),
            ("SIGNAL_TELEMETRY", signal_telemetry(), 8.2),
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

        optional_chunks: List[Tuple[str, str, float]] = [
            ("QUANTUM_PROJECTION", quantum_projection(), 7.4),
            ("AGENT_OPERATING_MODEL", agent_operating_model(), 7.3),
            ("HYPOTHESIS_LATTICE", hypothesis_lattice(), 7.2),
            ("RESPONSE_TUNING", response_tuning(), 7.1),
            ("ADVANCED_IDEAS", build_advanced_quantum_ai_ideas(), 7.35),
            ("LLAMA_LOCAL_PLAYBOOK", build_llama_local_playbook(), 7.25),
        ]
        ]
        if domain == "book_generator":
            book_chunks = [
                ("BOOK_BLUEPRINT", build_book_blueprint(), 8.5),
                ("BOOK_QUALITY_MATRIX", build_book_quality_matrix(), 8.3),
                ("BOOK_DELIVERY_SPEC", build_book_delivery_spec(), 8.2),
                ("REVOLUTIONARY_IDEAS", build_book_revolutionary_ideas(), 8.1),
                ("REVOLUTIONARY_IDEAS_V2", build_book_revolutionary_ideas_v2(), 8.05),
                ("BOOK_REVIEW_STACK", build_book_review_stack(), 8.0),
                ("PUBLISHING_POLISHER", build_publishing_polisher(), 7.95),
                ("SEMANTIC_CLARITY", build_semantic_clarity_stack(), 7.9),
                ("GENRE_MATRIX", build_genre_matrix(), 7.85),
                ("VOICE_READING_PLAN", build_voice_reading_plan(), 7.8),
                ("REVOLUTIONARY_DEPLOYMENTS", build_book_revolutionary_deployments(), 8.0),
                ("REVOLUTIONARY_DEPLOYMENTS_EXT", build_book_revolutionary_deployments_extended(), 7.95),
                ("REVOLUTIONARY_DEPLOYMENTS_SUPER", build_book_revolutionary_deployments_super(), 7.9),
            ]
            optional_chunks.extend(book_chunks)

        if risk >= 0.66 or quantum_gain >= 0.7:
            optional_chunks.append((
                "CONTAINMENT_PRIORITY",
                "CONTAINMENT PRIORITY:\n- Contain → Verify → Recover. Keep actions reversible.",
                7.6,
            ))
        if conf < 0.65:
            optional_chunks.append((
                "UNCERTAINTY_PROTOCOL",
                "UNCERTAINTY PROTOCOL:\n- Ask high-value questions first.\n- Use safe defaults when data missing.",
                7.55,
            ))

        chunks = base_chunks + optional_chunks
        required_titles = {"SYSTEM_HEADER", "STATE_METRICS", "DOMAIN_SPEC", "USER_CONTEXT", "OUTPUT_SCHEMA", "NONNEGOTIABLE_RULES"}

        base_count = len(base_chunks)
        # The target max chunk count scales with quantum_gain to surface
        # more advanced instructions when the system is "coherent."
        target_max = min(self.max_chunks, max(base_count, 10 + int(quantum_gain * 4)))
        required = [c for c in chunks if c[0] in required_titles]
        optional = [c for c in chunks if c[0] not in required_titles]
        optional.sort(key=lambda c: c[2], reverse=True)
        selected = required + optional
        selected = selected[:target_max]

        out: List[PromptChunk] = []
        for i, (title, txt, base_w) in enumerate(selected):
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
        quantum_gain = float(ctx.get("quantum_gain", 0.0))

        base_temp = 0.35 + 0.22 * (1.0 - risk)
        base_temp -= 0.14 * (1.0 - conf)
        base_temp -= 0.10 * min(1.0, vol)
        base_temp -= 0.08 * min(1.0, abs(drift))
        base_temp -= 0.12 * min(1.0, quantum_gain)
        temp = float(max(0.06, min(0.75, base_temp)))

        est_in = self._estimate_tokens(draft.render(with_rgb_tags=True))
        if est_in > 2600:
            out = 256
        elif est_in > 1900:
            out = 384
        else:
            out = 500 + int(450 * min(1.0, risk + (1.0 - conf)))
            out = int(out * (0.88 + 0.24 * (1.0 - quantum_gain)))
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
        with_rgb_tags: bool = True,
        signal_summary: Optional[Dict[str, Any]] = None,
    ) -> PromptPlan:
        # Orchestrate chunk building, agent actions, and length guarding into a
        # final PromptPlan with metadata for debugging/telemetry.
        chunks = self.chunker.build(
            domain=domain,
            metrics=metrics,
            ceb_sig=ceb_sig,
            base_rgb=base_rgb,
            signal_summary=signal_summary,
        )
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
            "signals": signal_summary or {},
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
        url = f"{self.base_url}/responses"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "input": prompt,
            "input": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
            "store": False,
        }

        last_err: Optional[Exception] = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(int(retries)):
                try:
                    r = client.post(url, headers=headers, json=payload)
                    if r.status_code >= 400:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                    j = r.json()
                    out = self._extract_text_from_responses_api(j)
                    if out:
                        return out
                    return json.dumps(j, ensure_ascii=False)
                except Exception as e:
                    last_err = e
                    time.sleep((2 ** attempt) * 0.6)
        raise RuntimeError(f"OpenAI call failed: {last_err}")

    @staticmethod
    def _extract_text_from_responses_api(j: Dict[str, Any]) -> str:
        texts: List[str] = []
        for item in j.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []) or []:
                if part.get("type") != "output_text":
                    continue
                t = part.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t)
            if item.get("type") == "message":
                for part in item.get("content", []) or []:
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t)
        return "\n".join(texts).strip()

    def chat_longform(self, prompt: str, temperature: float, max_tokens: int, retries: int = 3) -> str:
        chunks = chunk_text_for_longform(prompt, max_chars=3800)
        outputs = []
        for idx, chunk in enumerate(chunks, start=1):
            header = f"[CHUNK {idx}/{len(chunks)}]\n"
            outputs.append(self.chat(header + chunk, temperature=temperature, max_tokens=max_tokens, retries=retries))
        return concat_longform(outputs)


class HttpxGrokClient:
    def __init__(self, api_key: str, base_url: str = GROK_BASE_URL, model: str = GROK_MODEL, timeout_s: float = 60.0):
        if not api_key:
            raise RuntimeError("GROK_API_KEY not set.")
        if not base_url:
            raise RuntimeError("GROK_BASE_URL not set.")
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
        raise RuntimeError(f"Grok call failed: {last_err}")


class HttpxGeminiClient:
    def __init__(self, api_key: str, base_url: str = GEMINI_BASE_URL, model: str = GEMINI_MODEL, timeout_s: float = 60.0):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        if not base_url:
            raise RuntimeError("GEMINI_BASE_URL not set.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(timeout_s)

    def chat(self, prompt: str, temperature: float, max_tokens: int, retries: int = 3) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_tokens)},
        }
        last_err: Optional[Exception] = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(int(retries)):
                try:
                    r = client.post(url, json=payload)
                    if r.status_code >= 400:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                    j = r.json()
                    return j["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception as e:
                    last_err = e
                    time.sleep((2 ** attempt) * 0.6)
        raise RuntimeError(f"Gemini call failed: {last_err}")


def download_llama3_model(target_path: str) -> None:
    if not LLAMA3_MODEL_URL or not LLAMA3_MODEL_SHA256:
        raise RuntimeError("LLAMA3_MODEL_URL or LLAMA3_MODEL_SHA256 not set.")
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", LLAMA3_MODEL_URL, timeout=120.0) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        hasher = hashlib.sha256()
        with target.open("wb") as f:
            for chunk in r.iter_bytes():
                if not chunk:
                    continue
                hasher.update(chunk)
                f.write(chunk)
    if hasher.hexdigest().lower() != LLAMA3_MODEL_SHA256.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError("Llama3 model hash mismatch.")


def download_llama3_2_model(target_path: str) -> None:
    if not LLAMA3_2_MODEL_URL or not LLAMA3_2_MODEL_SHA256:
        raise RuntimeError("LLAMA3_2_MODEL_URL or LLAMA3_2_MODEL_SHA256 not set.")
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", LLAMA3_2_MODEL_URL, timeout=120.0) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        hasher = hashlib.sha256()
        with target.open("wb") as f:
            for chunk in r.iter_bytes():
                if not chunk:
                    continue
                hasher.update(chunk)
                f.write(chunk)
    if hasher.hexdigest().lower() != LLAMA3_2_MODEL_SHA256.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError("Llama3.2 model hash mismatch.")


def encrypt_llama3_model(src_path: str, dst_path: str) -> None:
    if not LLAMA3_AES_KEY_B64:
        raise RuntimeError("LLAMA3_AES_KEY_B64 not set.")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = base64.b64decode(LLAMA3_AES_KEY_B64.encode("utf-8"))
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    data = Path(src_path).read_bytes()
    encrypted = aesgcm.encrypt(nonce, data, None)
    Path(dst_path).write_bytes(nonce + encrypted)


def decrypt_llama3_model(src_path: str, dst_path: str) -> None:
    if not LLAMA3_AES_KEY_B64:
        raise RuntimeError("LLAMA3_AES_KEY_B64 not set.")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = base64.b64decode(LLAMA3_AES_KEY_B64.encode("utf-8"))
    aesgcm = AESGCM(key)
    blob = Path(src_path).read_bytes()
    nonce = blob[:12]
    data = blob[12:]
    decrypted = aesgcm.decrypt(nonce, data, None)
    Path(dst_path).write_bytes(decrypted)

    def chat_longform(self, prompt: str, temperature: float, max_tokens: int, retries: int = 3) -> str:
        chunks = chunk_text_for_longform(prompt, max_chars=3800)
        outputs = []
        for idx, chunk in enumerate(chunks, start=1):
            header = f"[CHUNK {idx}/{len(chunks)}]\n"
            outputs.append(self.chat(header + chunk, temperature=temperature, max_tokens=max_tokens, retries=retries))
        return concat_longform(outputs)


class HttpxGrokClient:
    def __init__(self, api_key: str, base_url: str = GROK_BASE_URL, model: str = GROK_MODEL, timeout_s: float = 60.0):
        if not api_key:
            raise RuntimeError("GROK_API_KEY not set.")
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
        raise RuntimeError(f"Grok call failed: {last_err}")


class HttpxGeminiClient:
    def __init__(self, api_key: str, base_url: str = GEMINI_BASE_URL, model: str = GEMINI_MODEL, timeout_s: float = 60.0):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        if not base_url:
            raise RuntimeError("GEMINI_BASE_URL not set.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(timeout_s)

    def chat(self, prompt: str, temperature: float, max_tokens: int, retries: int = 3) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_tokens)},
        }
        last_err: Optional[Exception] = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(int(retries)):
                try:
                    r = client.post(url, json=payload)
                    if r.status_code >= 400:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                    j = r.json()
                    return j["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception as e:
                    last_err = e
                    time.sleep((2 ** attempt) * 0.6)
        raise RuntimeError(f"Gemini call failed: {last_err}")


def download_llama3_model(target_path: str) -> None:
    if not LLAMA3_MODEL_URL or not LLAMA3_MODEL_SHA256:
        raise RuntimeError("LLAMA3_MODEL_URL or LLAMA3_MODEL_SHA256 not set.")
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", LLAMA3_MODEL_URL, timeout=120.0) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        hasher = hashlib.sha256()
        with target.open("wb") as f:
            for chunk in r.iter_bytes():
                if not chunk:
                    continue
                hasher.update(chunk)
                f.write(chunk)
    if hasher.hexdigest().lower() != LLAMA3_MODEL_SHA256.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError("Llama3 model hash mismatch.")


def download_llama3_2_model(target_path: str) -> None:
    if not LLAMA3_2_MODEL_URL or not LLAMA3_2_MODEL_SHA256:
        raise RuntimeError("LLAMA3_2_MODEL_URL or LLAMA3_2_MODEL_SHA256 not set.")
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", LLAMA3_2_MODEL_URL, timeout=120.0) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        hasher = hashlib.sha256()
        with target.open("wb") as f:
            for chunk in r.iter_bytes():
                if not chunk:
                    continue
                hasher.update(chunk)
                f.write(chunk)
    if hasher.hexdigest().lower() != LLAMA3_2_MODEL_SHA256.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError("Llama3.2 model hash mismatch.")


def encrypt_llama3_model(src_path: str, dst_path: str) -> None:
    if not LLAMA3_AES_KEY_B64:
        raise RuntimeError("LLAMA3_AES_KEY_B64 not set.")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = base64.b64decode(LLAMA3_AES_KEY_B64.encode("utf-8"))
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    data = Path(src_path).read_bytes()
    encrypted = aesgcm.encrypt(nonce, data, None)
    Path(dst_path).write_bytes(nonce + encrypted)


def decrypt_llama3_model(src_path: str, dst_path: str) -> None:
    if not LLAMA3_AES_KEY_B64:
        raise RuntimeError("LLAMA3_AES_KEY_B64 not set.")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = base64.b64decode(LLAMA3_AES_KEY_B64.encode("utf-8"))
    aesgcm = AESGCM(key)
    blob = Path(src_path).read_bytes()
    nonce = blob[:12]
    data = blob[12:]
    decrypted = aesgcm.decrypt(nonce, data, None)
    Path(dst_path).write_bytes(decrypted)


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
        self.signal_pipeline = SignalPipeline()
        self.last_signals: Optional[SystemSignals] = None
        self._plan_cache: Dict[str, Tuple[Tuple[Any, ...], PromptPlan]] = {}

        self.last_scans: Dict[str, DomainScan] = {}
        self.focus_idx = 0
        self._lock = threading.Lock()

    def scan_once(self) -> Dict[str, DomainScan]:
        # One full scan: sample signals, build entropy/lattice, evolve CEBs,
        # compute per-domain metrics, and assemble prompt plans. This is the
        # main heartbeat of the system.
        raw_signals = SystemSignals.sample()
        signals = self.signal_pipeline.update(raw_signals)
        self.last_signals = signals
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
        if sig.get("entropy", 0.0) < 3.0:
            st = self.ceb.evolve(st, entropy_blob=ent_blob, steps=90, drift_bias=global_bias, chroma_gain=1.25)
            p = self.ceb.probs(st)
            sig = self.ceb.signature(st, k=12)

        scans: Dict[str, DomainScan] = {}

        self.memory.decay()

        for d in self.domains:
            sl = _domain_slice(d, p)
            d_entropy = domain_entropy_from_slice(sl)
            self.memory.update(d, d_entropy)

            drift = self.memory.weighted_drift(d)
            conf = self.memory.confidence(d)
            vol = self.memory.stats(d)["volatility"]
            shock = self.memory.shock(d)
            anomaly = self.memory.anomaly(d)

            base_risk = domain_risk_from_ceb(d, p)
            risk = apply_cross_domain_bias(d, base_risk, self.memory)
            risk = adjust_risk_by_confidence(risk, conf, vol)
            risk = adjust_risk_by_instability(risk, shock, anomaly)
            status = status_from_risk(risk)

            metrics = {
                "risk": float(risk),
                "drift": float(drift),
                "confidence": float(conf),
                "volatility": float(vol),
                "shock": float(shock),
                "anomaly": float(anomaly),
            }
            status = status_from_risk(risk)

            metrics = {"risk": float(risk), "drift": float(drift), "confidence": float(conf), "volatility": float(vol)}
            quantum_summary = build_quantum_advancements(signals, sig, metrics, loops=5)
            metrics["quantum_gain"] = float(quantum_summary.get("quantum_gain", 0.0))
            metrics["quantum_summary"] = quantum_summary

            top_colors = [tuple(t.get("rgb", [])) for t in sig.get("top", [])[:6]]
            sig_key = tuple(int(c) for rgb in top_colors for c in rgb)
            key = (
                round(metrics["risk"], 4),
                round(metrics["drift"], 4),
                round(metrics["confidence"], 4),
                round(metrics["volatility"], 4),
                round(metrics.get("shock", 0.0), 4),
                round(metrics.get("anomaly", 0.0), 4),
                round(metrics.get("quantum_gain", 0.0), 4),
                sig_key,
            )
            plan = None
            cache = self._plan_cache.get(d)
            if cache and cache[0] == key:
                plan = cache[1]
            if plan is None:
                signal_summary = {
                    "cpu": round(signals.cpu_percent, 2),
                    "disk": round(signals.disk_percent, 2),
                    "ram_ratio": round(signals.ram_ratio, 3),
                    "net_rate": int(signals.net_rate),
                    "uptime_s": int(signals.uptime_s),
                    "proc": int(signals.proc_count),
                    "jitter": round(signals.cpu_jitter + signals.disk_jitter, 3),
                    "quantum_gain": round(metrics["quantum_gain"], 4),
                }
                plan = self.orch.build_plan(
                    domain=d,
                    metrics=metrics,
                    ceb_sig=sig,
                    base_rgb=base_rgb,
                    max_prompt_chars=MAX_PROMPT_CHARS,
                    with_rgb_tags=True,
                    signal_summary=signal_summary,
                )
                self._plan_cache[d] = (key, plan)

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

    def get_last_signals(self) -> Optional[SystemSignals]:
        return self.last_signals


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
        self._last_ai_time: Dict[str, float] = {}

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"{ts} {msg}"
        with self._lock:
            self.logs.append(line)
            self.logs = self.logs[-LOG_BUFFER_LINES:]

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
        # Render a full frame: dashboard, signature, prompt panel, logs,
        # and a signal status bar. Keep operations small to stay within
        # the TUI refresh cadence.
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        dash_h = 9
        log_h = 6
        sig_h = 10
        mid_h = h - dash_h - log_h - 1
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

        self._draw_logs(stdscr, logs_copy, y=h - log_h - 1, x=2, height=log_h, width=w - 4)
        self._draw_statusbar(stdscr, y=h - 1, width=w)
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

    def _draw_statusbar(self, stdscr, y: int, width: int) -> None:
        sig = self.sys.get_last_signals()
        if not sig:
            return
        msg = (
            f"CPU {sig.cpu_percent:5.1f}% | DISK {sig.disk_percent:5.1f}% | "
            f"RAM {sig.ram_ratio * 100:5.1f}% | NET {sig.net_rate:8.0f}B/s | "
            f"UP {sig.uptime_s:8.0f}s | PROC {sig.proc_count:5d} | JIT {sig.cpu_jitter + sig.disk_jitter:5.2f}"
        )
        stdscr.addstr(y, 0, msg[: max(0, width - 1)], curses.A_REVERSE)

    def _run_ai_for_focus(self) -> None:
        # AI calls are optional and rate-limited. We enforce cooldown to
        # prevent repeated API calls from overwhelming the user or quota.
        focus = self.sys.get_focus_domain()
        with self._lock:
            s = self.scans.get(focus)

        if not s or not s.prompt_plan:
            self.log("No prompt available.")
            return
        if not OPENAI_API_KEY:
            self.log("OPENAI_API_KEY not set.")
            return
        last = self._last_ai_time.get(focus, 0.0)
        if time.time() - last < AI_COOLDOWN_SECONDS:
            wait_s = int(AI_COOLDOWN_SECONDS - (time.time() - last))
            self.log(f"AI cooldown active ({wait_s}s remaining).")
            return

        self.log(f"AI run start for {focus}...")
        self._last_ai_time[focus] = time.time()

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
    domains = DEFAULT_DOMAINS
    if BOOK_TITLE:
        # When a book title is provided, focus on the book generator domain.
        domains = ["book_generator"]
    sys = RGNCebSystem(domains=domains)
    tui = AdvancedTUI(sys)
    tui.run()


if __name__ == "__main__":
    main()
