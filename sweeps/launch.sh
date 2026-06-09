#!/usr/bin/env bash
# Helper: log into wandb + huggingface, create a sweep, launch an agent.
#
# Usage:
#   ./sweeps/launch.sh <sweep.yaml> [--count N]
#
# Expects WANDB_API_KEY and HF_TOKEN in the environment. Refuses to run if either
# is missing — better to fail loudly than to launch silently-anonymous runs.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <sweep.yaml> [--count N]" >&2
  exit 2
fi

SWEEP_YAML="$1"
shift

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set in the environment." >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set in the environment." >&2
  exit 1
fi
if [[ ! -f "$SWEEP_YAML" ]]; then
  echo "ERROR: sweep YAML not found: $SWEEP_YAML" >&2
  exit 1
fi

echo ">> Logging in to wandb..."
wandb login --relogin "$WANDB_API_KEY"

echo ">> Logging in to Hugging Face..."
# Newer huggingface_hub uses 'hf auth login'; older uses 'huggingface-cli login'.
if command -v hf >/dev/null 2>&1; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential || true
else
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo ">> Creating sweep from $SWEEP_YAML ..."
# `wandb sweep` writes the sweep ID to stderr; capture both streams and grep it out.
SWEEP_OUTPUT="$(wandb sweep "$SWEEP_YAML" 2>&1 | tee /dev/stderr)"
SWEEP_ID="$(echo "$SWEEP_OUTPUT" | grep -oE 'mcclain/[^[:space:]]+/[a-z0-9]+' | tail -n1)"

if [[ -z "$SWEEP_ID" ]]; then
  echo "ERROR: could not parse sweep ID from wandb output." >&2
  exit 1
fi

echo ">> Sweep ID: $SWEEP_ID"
echo ">> Launching agent..."
exec wandb agent "$SWEEP_ID" "$@"
