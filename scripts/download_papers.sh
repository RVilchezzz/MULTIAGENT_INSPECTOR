#!/usr/bin/env bash
# Download the main reference PDFs from arXiv into papers/
set -euo pipefail
cd "$(dirname "$0")/../papers"
declare -A P=(
  [2601.13671_orchestration_multi_agent_systems]=2601.13671
  [2610.12310_SiGMA]=2610.12310
  [2410.09453_MMAD]=2410.09453
  [2503.13657_MAST_why_multiagent_fail]=2503.13657
  [2402.01680_LLM_multiagents_survey]=2402.01680
  [2305.14325_multiagent_debate]=2305.14325
  [2312.13010_AgentCoder]=2312.13010
  [2603.13779_AD-Copilot]=2603.13779
  [2501.15795_Echo]=2501.15795
  [2508.10104_DINOv3]=2508.10104
  [2403.12580_Real-IAD]=2403.12580
)
for name in "${!P[@]}"; do
  f="$name.pdf"
  [ -f "$f" ] && { echo "ok   $f"; continue; }
  echo "get  $f"; curl -fsSL -o "$f" "https://arxiv.org/pdf/${P[$name]}" || echo "failed: ${P[$name]}"
  sleep 3   # arXiv rate-limits requests
done
