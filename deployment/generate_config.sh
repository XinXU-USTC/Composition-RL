#!/bin/bash
# generate_config_from_running.sh

source nodes_config.sh

CONFIG_FILE="generated_vllm_config.yaml"

echo "Scanning for running vLLM instances..."

# Detect running instances
declare -a RUNNING_PORTS
declare -a RUNNING_NODES

while IFS= read -r node; do
    for port in {8000..8010}; do
        if curl -s --max-time 2 http://$node:$port/health > /dev/null 2>&1; then
            echo "  Found: $node:$port"
            RUNNING_PORTS+=($port)
            RUNNING_NODES+=($node)
        fi
    done
done < pssh.hosts

if [ ${#RUNNING_PORTS[@]} -eq 0 ]; then
    echo "No running vLLM instances found!"
    exit 1
fi

# Get unique ports
UNIQUE_PORTS=($(echo "${RUNNING_PORTS[@]}" | tr ' ' '\n' | sort -u))

echo ""
echo "Found ${#RUNNING_NODES[@]} running instances on ports: ${UNIQUE_PORTS[*]}"
echo ""

# Generate config
cat > $CONFIG_FILE << EOF
# Auto-generated vLLM configuration
# Generated at: $(date)
# Detected running instances: ${#RUNNING_NODES[@]}

resp_urls:
EOF

for ((i=0; i<${#RUNNING_NODES[@]}; i++)); do
    node=${RUNNING_NODES[$i]}
    port=${RUNNING_PORTS[$i]}
    echo "  - \"http://$node:$port/v1\"" >> $CONFIG_FILE
done

cat >> $CONFIG_FILE << EOF

resp_server_names:
EOF

for ((i=0; i<${#RUNNING_NODES[@]}; i++)); do
    echo "  - \"$MODEL\"" >> $CONFIG_FILE
done

cat >> $CONFIG_FILE << EOF

resp_api_keys:
EOF

for ((i=0; i<${#RUNNING_NODES[@]}; i++)); do
    echo "  - \"test\"" >> $CONFIG_FILE
done

echo "Generated: $CONFIG_FILE"
echo ""
cat $CONFIG_FILE