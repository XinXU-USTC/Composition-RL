#!/bin/bash

source nodes_config.sh

echo "Stopping all vLLM instances across cluster..."

# Stop all vLLM processes in parallel
pssh -h pssh.hosts -i -t 30 \
    "pkill -f 'vllm serve' && echo '✓ Stopped vLLM processes' || echo '✗ No vLLM processes found'"

echo ""
echo "All instances stopped."