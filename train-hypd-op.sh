#!/bin/bash

cd /workspace/Megatron-LM/
python3 setup.py build_ext --inplace
pip install --no-build-isolation .

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
CHECKPOINT_PATH=${1:-"/workspace/checkpoints/llama3_fp8"}
TENSORBOARD_LOGS_PATH=${2:-"/workspace/tensorboard_logs/llama3_fp8"}
TOKENIZER_ARG=${3:-"MOCK"}
DATA_ARG=${4:-"MOCK"}

DATA_CACHE_PATH="/workspace/built_datasets/llama3_fp8"

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"
mkdir -p "$(dirname "$DATA_CACHE_PATH")"

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

TP_SIZE=4
CP_SIZE=2     
PP_SIZE=1     
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
NUM_LAYERS=4  
DTYPE="fp8"
SEQ_LENGTH=512
MAX_POSITION_EMBEDDINGS=512


MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size 256
    --ffn-hidden-size 512
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 8
    --kv-channels 16
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000 
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p 
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    # --no-gradient-accumulation-fusion
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 50000
    --lr-decay-samples 40000
    --lr-warmup-samples 10000
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4      # Specific to decoupled AdamW, ensure optimizer is compatible
    --decoupled-min-lr 4.5e-5  # Specific to decoupled AdamW
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
    --exit-duration-in-mins 235 
)

# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --sequence-parallel
)

DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256" 
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        "--vocab-size 128256"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH" 
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

echo "HYPERPOD TRAINING OP NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "HYPERPOD TRAINING OP NNODES: ${NNODES}"

# torchrun ${DISTRIBUTED_ARGS[@]} \
exec hyperpodrun \
    --nnodes=${NNODES} --nproc-per-node=${NPROC_PER_NODE} \
    --server-host=0.0.0.0 --server-port=8080 \
    --tee=3 --log_dir=/workspace/hyperpodrun-logs \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x