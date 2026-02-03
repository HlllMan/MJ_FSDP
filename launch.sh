torchrun \
    --nnodes=2 \
    --node_rank=${1} \
    --master_addr=${2} \
    --master_port=${3:-29500} \
    --nproc_per_node=8 \
    TP.py