#!/bin/bash

#================================================
# Cluster Configuration
#================================================

export NODE_IP_LIST="${NODE_IP_LIST:-192.168.1.101:8,192.168.1.102:8,192.168.1.103:8,192.168.1.104:8}"

# Generate hostfile and pssh.hosts
echo $NODE_IP_LIST > env.txt 2>&1
sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" > "hostfile"
sed "s/:.//g" env.txt | sed "s/,/\n/g" > "pssh.hosts"
main_ip=$(head -n 1 pssh.hosts)

#================================================
# GPU Configuration
#================================================

GPUS_PER_NODE=$(echo $NODE_IP_LIST | cut -d',' -f1 | cut -d':' -f2)
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Set Tensor Parallel Size
TP_SIZE=4

# Auto-calculate number of instances per node
INSTANCES_PER_NODE=$((GPUS_PER_NODE / TP_SIZE))

# Validate TP_SIZE
if [ $((GPUS_PER_NODE % TP_SIZE)) -ne 0 ]; then
    echo "ERROR: TP_SIZE ($TP_SIZE) must evenly divide GPUS_PER_NODE ($GPUS_PER_NODE)"
    echo "Valid TP_SIZE values for $GPUS_PER_NODE GPUs: 1, 2, 4, 8"
    exit 1
fi

#================================================
# Port Configuration
#================================================

# Port range to search
PORT_RANGE_START=8000
PORT_RANGE_END=9000
PREFERRED_START_PORT=8000

# Will be populated by find_free_ports()
PORTS=()
GPU_ASSIGNMENTS=()

#================================================
# vLLM Model Configuration
#================================================

MODEL="Qwen/Qwen2.5-32B-Instruct"
MAX_MODEL_LEN=20480
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=256

#================================================
# Environment Configuration
#================================================

#CONDA_ENV="vllm_env"
ROOT_PATH=`pwd`

#================================================
# Functions
#================================================

# Check if port is free on a remote node
check_port_free() {
    local node=$1
    local port=$2
    
    # Check if port is in use
    ssh -o ConnectTimeout=5 $node "! ss -tuln | grep -q ':${port} '" 2>/dev/null
    return $?
}

# Test if port is reachable from master node
test_port_connectivity() {
    local node=$1
    local port=$2
    local timeout=3
    
    # Try to connect to port
    timeout $timeout bash -c "cat < /dev/null > /dev/tcp/$node/$port" 2>/dev/null
    return $?
}

# Find N consecutive free ports on all nodes
find_free_ports() {
    local num_ports=$1
    local start_port=${2:-$PREFERRED_START_PORT}
    
    echo "Scanning for $num_ports free ports starting from $start_port..."
    
    local found_ports=()
    local current_port=$start_port
    
    while [ ${#found_ports[@]} -lt $num_ports ] && [ $current_port -lt $PORT_RANGE_END ]; do
        local port_free_on_all=true
        
        # Check if port is free on ALL nodes
        while IFS= read -r node; do
            if ! check_port_free $node $current_port; then
                port_free_on_all=false
                break
            fi
        done < pssh.hosts
        
        if [ "$port_free_on_all" = true ]; then
            found_ports+=($current_port)
            echo "  ✓ Port $current_port is free on all nodes"
        else
            echo "  ✗ Port $current_port is in use on at least one node"
        fi
        
        ((current_port++))
    done
    
    if [ ${#found_ports[@]} -lt $num_ports ]; then
        echo "ERROR: Could not find $num_ports free ports in range $start_port-$PORT_RANGE_END"
        return 1
    fi
    
    # Export found ports
    PORTS=("${found_ports[@]}")
    return 0
}

# Test firewall and connectivity for all ports
test_all_ports_connectivity() {
    echo ""
    echo "Testing port connectivity from master node..."
    
    local all_reachable=true
    
    while IFS= read -r node; do
        echo "  Testing node: $node"
        for port in "${PORTS[@]}"; do
            # First check if port allows incoming connections (firewall test)
            if ssh -o ConnectTimeout=5 $node "timeout 2 nc -l $port" 2>/dev/null &
            then
                local ssh_pid=$!
                sleep 1
                
                # Try to connect from master
                if timeout 2 bash -c "echo test > /dev/tcp/$node/$port" 2>/dev/null; then
                    echo "    ✓ Port $port is reachable"
                    ssh $node "kill $ssh_pid 2>/dev/null" || true
                else
                    echo "    ✗ Port $port is NOT reachable (firewall issue?)"
                    all_reachable=false
                    ssh $node "kill $ssh_pid 2>/dev/null" || true
                fi
            fi
        done
    done < pssh.hosts
    
    if [ "$all_reachable" = false ]; then
        echo ""
        echo "⚠ WARNING: Some ports are not reachable. You may need to:"
        echo "  1. Open firewall ports on nodes"
        echo "  2. Check security groups (cloud environments)"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    
    return 0
}

# Generate GPU assignments
generate_gpu_assignments() {
    GPU_ASSIGNMENTS=()
    for ((i=0; i<$INSTANCES_PER_NODE; i++)); do
        start_gpu=$((i * TP_SIZE))
        end_gpu=$((start_gpu + TP_SIZE - 1))
        
        gpu_ids=""
        for ((j=start_gpu; j<=end_gpu; j++)); do
            if [ -z "$gpu_ids" ]; then
                gpu_ids="$j"
            else
                gpu_ids="$gpu_ids,$j"
            fi
        done
        GPU_ASSIGNMENTS+=("$gpu_ids")
    done
}

#================================================
# Main Configuration Logic
#================================================

# Find free ports
if ! find_free_ports $INSTANCES_PER_NODE $PREFERRED_START_PORT; then
    echo "Failed to find free ports. Exiting."
    exit 1
fi

# Generate GPU assignments
generate_gpu_assignments

# Export variables
export MODEL TP_SIZE MAX_MODEL_LEN GPU_MEMORY_UTIL MAX_NUM_SEQS ROOT_PATH
export GPUS_PER_NODE INSTANCES_PER_NODE
export PORTS_STR="${PORTS[*]}"
export GPU_ASSIGNMENTS_STR="${GPU_ASSIGNMENTS[*]}"

#================================================
# Display Configuration
#================================================

echo ""
echo "======================================"
echo "  vLLM Cluster Configuration"
echo "======================================"
echo ""
echo "Cluster Info:"
echo "  Main IP: $main_ip"
echo "  Total Nodes: $(cat pssh.hosts | wc -l)"
echo "  GPUs per Node: $GPUS_PER_NODE"
echo ""
echo "Deployment Strategy:"
echo "  Tensor Parallel Size: $TP_SIZE"
echo "  Instances per Node: $INSTANCES_PER_NODE"
echo "  Total Instances: $(($(cat pssh.hosts | wc -l) * INSTANCES_PER_NODE))"
echo ""
echo "Port Assignments (Auto-discovered):"
for ((i=0; i<${#PORTS[@]}; i++)); do
    echo "  Instance $((i+1)): Port ${PORTS[$i]} → GPUs [${GPU_ASSIGNMENTS[$i]}]"
done
echo ""
echo "Model Configuration:"
echo "  Model: $MODEL"
echo "  Max Sequence Length: $MAX_MODEL_LEN"
echo "  Max Concurrent Sequences: $MAX_NUM_SEQS"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo ""
echo "======================================"
