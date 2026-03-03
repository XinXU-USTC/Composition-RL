#!/bin/bash

set -e

# Load configuration
source nodes_config.sh

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  vLLM Cluster Deployment${NC}"
echo -e "${GREEN}=====================================${NC}\n"

# Validate configuration
if [ ${#PORTS[@]} -eq 0 ]; then
    echo -e "${RED}ERROR: No ports configured.${NC}"
    exit 1
fi

# Test basic node connectivity (NOT port connectivity)
echo -e "${YELLOW}Testing node connectivity...${NC}"
all_reachable=true
while IFS= read -r node; do
    if ping -c 1 -W 2 $node >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $node is reachable"
    else
        echo -e "  ${RED}✗${NC} $node is NOT reachable"
        all_reachable=false
    fi
done < pssh.hosts

if [ "$all_reachable" = false ]; then
    echo -e "\n${RED}Some nodes are not reachable. Aborting deployment.${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ All nodes are reachable${NC}\n"

# Copy deployment script to all nodes
echo -e "${YELLOW}Copying deployment script to all nodes...${NC}"
pssh -h pssh.hosts -i -t 30 "mkdir -p ~/vllm_deployment"
pscp.pssh -h pssh.hosts -t 30 remote_deploy_vllm.sh ~/vllm_deployment/

# Deploy all instances on all nodes
echo -e "\n${YELLOW}Deploying ${INSTANCES_PER_NODE} vLLM instances per node...${NC}\n"

for ((instance_idx=0; instance_idx<$INSTANCES_PER_NODE; instance_idx++)); do
    port=${PORTS[$instance_idx]}
    gpu_ids=${GPU_ASSIGNMENTS[$instance_idx]}
    
    echo -e "${GREEN}Deploying Instance $((instance_idx+1))/${INSTANCES_PER_NODE}:${NC}"
    echo -e "  Port: $port"
    echo -e "  GPUs: $gpu_ids"
    echo ""
    
    # Deploy on all nodes in parallel
    pssh -h pssh.hosts -i -t 120 \
        "export MODEL='$MODEL' TP_SIZE=$TP_SIZE MAX_MODEL_LEN=$MAX_MODEL_LEN \
         GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL MAX_NUM_SEQS=$MAX_NUM_SEQS CONDA_ENV=$CONDA_ENV \
         GPUS_PER_NODE=$GPUS_PER_NODE START_PORT=${PORTS[0]} && \
         bash ~/vllm_deployment/remote_deploy_vllm.sh $instance_idx $port $gpu_ids"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Instance $((instance_idx+1)) deployed successfully${NC}\n"
    else
        echo -e "${RED}✗ Instance $((instance_idx+1)) deployment failed on some nodes${NC}\n"
    fi
    
    sleep 5
done

# Wait for services to start
echo -e "${YELLOW}Waiting 60 seconds for services to initialize...${NC}"
for i in {60..1}; do
    echo -ne "  $i seconds remaining...\r"
    sleep 1
done
echo ""

# Health check with retry
echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}  Running Health Checks${NC}"
echo -e "${GREEN}=====================================${NC}\n"

declare -A failed_endpoints
max_retries=5

# Initialize all endpoints as failed
while IFS= read -r node; do
    for port in "${PORTS[@]}"; do
        failed_endpoints["$node:$port"]=1
    done
done < pssh.hosts

for retry in $(seq 1 $max_retries); do
    if [ $retry -gt 1 ]; then
        echo -e "${YELLOW}Retry $retry/$max_retries (waiting 10s)...${NC}\n"
        sleep 10
    fi
    
    while IFS= read -r node; do
        for port in "${PORTS[@]}"; do
            endpoint="$node:$port"
            
            # Skip if already healthy
            if [ -z "${failed_endpoints[$endpoint]}" ]; then
                continue
            fi
            
            # Test HTTP health
            if curl -s --max-time 5 http://$node:$port/health > /dev/null 2>&1; then
                echo -e "${GREEN}✓ $endpoint is healthy${NC}"
                unset failed_endpoints[$endpoint]
            else
                echo -e "${RED}✗ $endpoint not responding (attempt $retry/$max_retries)${NC}"
            fi
        done
    done < pssh.hosts
    
    # Break if all healthy
    if [ ${#failed_endpoints[@]} -eq 0 ]; then
        break
    fi
    
    echo ""
done

echo ""
if [ ${#failed_endpoints[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All vLLM instances are healthy!${NC}"
else
    echo -e "${RED}⚠ ${#failed_endpoints[@]} instance(s) failed health check:${NC}"
    for endpoint in "${!failed_endpoints[@]}"; do
        echo -e "${RED}  ✗ $endpoint${NC}"
    done
    echo ""
    echo -e "${YELLOW}To debug:${NC}"
    echo "  ./view_logs.sh 1 ${PORTS[0]}"
fi

# Test actual inference on one healthy endpoint
echo -e "\n${BLUE}=====================================${NC}"
echo -e "${BLUE}  Testing Inference${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Find a healthy endpoint
test_endpoint=""
while IFS= read -r node; do
    for port in "${PORTS[@]}"; do
        if [ -z "${failed_endpoints[$node:$port]}" ]; then
            test_endpoint="$node:$port"
            break 2
        fi
    done
done < pssh.hosts

if [ -z "$test_endpoint" ]; then
    echo -e "${RED}✗ No healthy endpoint found for inference test${NC}"
else
    test_node=$(echo $test_endpoint | cut -d: -f1)
    test_port=$(echo $test_endpoint | cut -d: -f2)
    
    echo "Testing inference on $test_endpoint..."
    
    response=$(curl -s --max-time 30 http://$test_node:$test_port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'$MODEL'",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 50,
            "temperature": 0.1
        }' 2>&1)
    
    if echo "$response" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        content=$(echo "$response" | jq -r '.choices[0].message.content')
        echo -e "${GREEN}✓ Inference test successful${NC}"
        echo -e "  Response: ${content:0:100}..."
    else
        echo -e "${RED}✗ Inference test failed${NC}"
        echo -e "  Response: ${response:0:200}..."
    fi
fi

# Generate deployment summary
node_count=$(cat pssh.hosts | wc -l)
total_instances=$((node_count * INSTANCES_PER_NODE))
healthy_count=$((total_instances - ${#failed_endpoints[@]}))

echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}  Deployment Summary${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "  Nodes: $node_count"
echo -e "  Instances per Node: $INSTANCES_PER_NODE"
echo -e "  Total Instances: $total_instances"
echo -e "  Healthy Instances: ${GREEN}$healthy_count${NC}"
if [ ${#failed_endpoints[@]} -gt 0 ]; then
    echo -e "  Failed Instances: ${RED}${#failed_endpoints[@]}${NC}"
fi
echo -e "  Tensor Parallel Size: $TP_SIZE"
echo -e "  Total GPUs Used: $((healthy_count * TP_SIZE))"
echo -e "  Ports Used: ${PORTS[*]}"
echo ""

# Generate YAML config (only healthy endpoints)
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  YAML Configuration${NC}"
echo -e "${GREEN}=====================================${NC}\n"

CONFIG_FILE="generated_vllm_config.yaml"
cat > $CONFIG_FILE << EOF
# Auto-generated vLLM configuration
# Generated at: $(date)
# Healthy instances: $healthy_count/$total_instances
# Ports: ${PORTS[*]}

resp_urls:
EOF

while IFS= read -r node; do
    for port in "${PORTS[@]}"; do
        endpoint="$node:$port"
        # Only include healthy endpoints
        if [ -z "${failed_endpoints[$endpoint]}" ]; then
            echo "  - \"http://$node:$port/v1\"" | tee -a $CONFIG_FILE
        fi
    done
done < pssh.hosts

cat >> $CONFIG_FILE << EOF

resp_server_names:
EOF

for ((i=1; i<=$healthy_count; i++)); do
    echo "  - \"$MODEL\"" | tee -a $CONFIG_FILE
done

cat >> $CONFIG_FILE << EOF

resp_api_keys:
EOF

for ((i=1; i<=$healthy_count; i++)); do
    echo "  - \"test\"" | tee -a $CONFIG_FILE
done

echo ""
echo -e "${GREEN}Configuration saved to: $CONFIG_FILE${NC}"
echo -e "${GREEN}Deployment complete!${NC}\n"

echo -e "${BLUE}Next steps:${NC}"
echo "  1. Monitor: ./monitor_cluster.sh"
echo "  2. Test: ./test_vllm_simple.sh"
echo "  3. Check status: ./check_vllm_status.sh"
echo ""