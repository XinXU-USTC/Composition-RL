set -x
# set up env
source /path/to/anaconda3/bin/activate your_env_name
# wandb key
export WANDB_API_KEY="your-key"

# NCCL-realted
export VLLM_ATTENTION_BACKEND=FLASH_ATTN #You could also switch to XFORMERS
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

#Multi-Nodes Settings
PET_MASTER_PORT=6379
# Replace $INDEX with your own node index variable
# Replace VC_MASTER_HOSTS with your own mater address variable

if [[ "$INDEX" == "0" ]];then
  echo "master, node_rank: $INDEX"
  
  # 先停止可能存在的Ray进程
  ray stop
  
  # 启动Ray head节点
  ray start --head \
    --port=$PET_MASTER_PORT \
    --num-gpus=8 \
    --node-ip-address=$VC_MASTER_HOSTS
  
  echo "ray start master in port: $VC_MASTER_HOSTS:$PET_MASTER_PORT"
  # 等待Ray Dashboard启动
  sleep 10
else
  echo "worker, node_rank: $INDEX"
  
  ray stop
  
  ray start --address="$VC_MASTER_HOSTS:$PET_MASTER_PORT" \
    --num-gpus=8
  
  sleep 10
  echo "ray start worker join master $VC_MASTER_HOSTS:$PET_MASTER_PORT"
  sleep infinity
fi