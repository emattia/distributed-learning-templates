# Set env vars in the .env file.
# This setup depends specifically on the HF_TOKEN.
set -a            
source .env
set +a

# Download the model.
tune download meta-llama/Meta-Llama-3-8B \
    --output-dir /tmp/Meta-Llama-3-8B \
    --hf-token $HF_TOKEN

# Finetune the model on 1 node (VM or Pod) with 2 GPUs.
tune run --nproc_per_node 2 \
    full_finetune_distributed \
    --config llama3/8B_full