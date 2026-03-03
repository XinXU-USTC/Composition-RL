set -x

# NCCL related
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_DEBUG="INFO"

MODEL_PATH="DeepScaleR-1.5B-Preview"
DATATYPES=("aime")
TEMP=0.6
TOP_P=0.95
LEN=32768
N_SAMPLE=8
TOP_K=-1
TP=1
USE_LONG_T=False
ADAPTIVE=False
QWEN3=True
DBS=2048
N_NODE=1
# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --temperature)
            TEMP="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --length)
            LEN="$2"
            shift 2
            ;;
        --n)
            N_SAMPLE="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --n_node)
            N_NODE="$2"
            shift 2
            ;;
        --data_bs)
            DBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

OUTPUT_DIR="$MODEL_PATH/eval_results"  # Add default output directory
MAX_PROMPT_LENGTH=2048
LEN1=$(expr $MAX_PROMPT_LENGTH + $LEN + 1000)

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation_eval \
        trainer.nnodes=$N_NODE  \
        trainer.n_gpus_per_node=8 \
        data.path=data/eval/$DATA_TYPE \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}_t${TEMP}_topp${TOP_P}_topk${TOP_K}_len${LEN}_n${N_SAMPLE}_uselong${USE_LONG_T}_adaptive${ADAPTIVE}_qwen3raw${QWEN3}_qwen_template.json \
        data.n_samples=$N_SAMPLE \
        data.batch_size=$DBS \
        model.path=${MODEL_PATH} \
        rollout.temperature=$TEMP \
        rollout.response_length=$LEN \
        rollout.prompt_length=$MAX_PROMPT_LENGTH \
        rollout.top_k=$TOP_K \
        rollout.top_p=$TOP_P \
        rollout.gpu_memory_utilization=0.92 \
        rollout.tensor_model_parallel_size=$TP \
        rollout.max_num_batched_tokens=$LEN1 \
        rollout.max_model_len=$LEN1 \
        rollout.enable_chunked_prefill=False \
        +data.skip_format_reward=True \
        +data.data_source_key="data_source" \
        +data.reward_model_key="reward_model"
done
# +data.skip_format_reward=True是默认行为，跳过校验答案正确性时的格式检查，没<think>也没事
# nnodes增大，则可增大gpu_memory_utilization至0.9-0.95 
# 如遇OOM，一般减小gpu_memory_utilization即可