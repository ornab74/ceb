# RGN CEB System

RGN CEB System is a terminal-based meta-prompt generator that computes Color-Entanglement Bits (CEBs) from live system signals and uses them to build domain-specific prompt plans. The program focuses on entropy-driven state, producing structured meta-prompts for different domains while displaying a curses TUI dashboard. The app can optionally call the OpenAI Chat Completions API via httpx when configured.

## What this project does

- **Generates meta-prompts** rather than finished advice: the output is a structured prompt that instructs another model to create domain checklists or suggestions. The code intentionally avoids embedding domain countermeasures directly.
- **Uses live system signals** (CPU, RAM, disk, network) to seed entropy and build CEB states that influence prompt content.
- **Runs as a TUI dashboard** with live updates, per-domain metrics, and keyboard controls for inspection and AI invocation.

## Architecture overview

The system is organized into a few major subsystems:

1. **System signal sampling** collects metrics (CPU, RAM, disk, network, uptime, process count) and handles restricted environments gracefully.
2. **Entropy + lattice generation** builds an RGB entropy vector and a quantum-inspired lattice, then amplifies entropy via hashing.
3. **CEB engine** initializes and evolves a CEB state, producing probabilities, a signature, and color-weighted entropy values.
4. **Domain slicing and risk metrics** map probabilities into per-domain slices, compute drift/volatility/confidence, and apply cross-domain bias.
5. **Prompt orchestration** builds a `PromptPlan` from prompt chunks, then applies agent actions (prioritize/trim/temperature/max_tokens).
6. **TUI renderer** displays dashboard metrics, prompt content, logs, and optional AI output.
7. **Optional OpenAI client** sends a prompt plan to the Chat Completions API when credentials are provided.

## Features

- **Meta-prompt generator only**: Builds prompts that instruct another model to produce checklists or suggestions; it does not embed domain countermeasures directly.
- **Entropy-driven CEB engine**: Samples system signals, generates RGB entropy, evolves CEB states, and computes per-domain risk/entropy metrics.
- **Domain orchestration**: Slices CEB probabilities into domains, applies cross-domain bias, and creates a `PromptPlan` with adjustable temperature/tokens.
- **Interactive TUI**: Curses dashboard shows per-domain risk, drift, confidence, volatility, and prompt output; includes keyboard controls for navigation and toggles.
- **Optional OpenAI client**: Uses `httpx` to call Chat Completions when `OPENAI_API_KEY` is set.

## Requirements

Install dependencies from the pinned list:

```bash
pip install -r requirements.txt
```

## Configuration

Optional environment variables supported by `main.py`:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-3.5-turbo"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export RGN_TUI_REFRESH="0.75"
```

### Model and API options

- `OPENAI_MODEL` controls the model name sent to the Chat Completions API.
- `OPENAI_BASE_URL` can point at alternative OpenAI-compatible endpoints.
- If `OPENAI_API_KEY` is not set, the app will run the TUI without AI calls and log a warning when AI is requested.

## Run

```bash
python main.py
```

## Keyboard controls

- `Q`: Quit
- `TAB`: Cycle domain focus
- `P`: Toggle prompt preview
- `O`: Toggle AI output panel
- `R`: Force rescan
- `A`: Run AI for focused domain
- `C`: Toggle colorized view

## Domains and coupling

The default domains include: `road_risk`, `vehicle_security`, `home_security`, `medicine_compliance`, `hygiene`, and `data_security`. Some domains influence each other via coupling (for example, `vehicle_security` and `data_security`), which can bias risk metrics across related areas.

## Prompt generation details

Each prompt plan is composed of multiple chunks that are color-tagged and weighted. The orchestration layer can prioritize sections, trim length when the prompt grows too large, and tune temperature/max token output based on risk, drift, confidence, and volatility metrics.

## TUI layout

The dashboard is split into sections that show:

- A **domain table** with risk, drift, confidence, volatility, and CEB entropy.
- A **CEB signature** view of the top color-weighted probabilities.
- A **prompt panel** that displays either the meta-prompt or the last AI response.
- A **log panel** with recent status updates.

## Troubleshooting

- If the screen is blank or colors are missing, confirm your terminal supports curses color mode.
- If network counters are unavailable (common in restricted environments), the app will fall back to zeros for network metrics.
- If OpenAI calls fail, check the API key, base URL, and network access.

## Development notes

The system is built as a single `main.py` entry point with no external configuration files. For modifications, start by scanning the `SystemSignals`, `CEBEngine`, and `PromptOrchestrator` sections, then follow the TUI rendering pipeline in `AdvancedTUI`.
