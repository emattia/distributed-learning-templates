# requires HF_TOKEN env var --> @secret


# tune download meta-llama/Meta-Llama-3-8B --output-dir /tmp/Meta-Llama-3-8B --hf-token <HF_TOKEN>
# tune run full_finetune_single_device --config llama3/8B_full_single_device
# tune run --nproc_per_node 4 full_finetune_distributed --config llama3/8B_full
# tune run --nproc_per_node 2 lora_finetune_distributed --config llama3/8B_lora

# tune download meta-llama/Meta-Llama-3-70b --hf-token <> --output-dir /tmp/Meta-Llama-3-70b --ignore-patterns "original/consolidated*"
# tune run --nproc_per_node 8 lora_finetune_distributed --config recipes/configs/llama3/70B_lora.yaml

## others
# tune run --nproc_per_node 8 lora_finetune_distributed --config recipes/configs/llama3/70B_lora.yaml
# tune run --nproc_per_node 2 full_finetune_distributed --config llama2/7B_full
# tune run lora_finetune_single_device --config llama2/7B_lora_single_device

## config overrides
# tune run lora_finetune_single_device \
# --config llama2/7B_lora_single_device \
# batch_size=8 \
# enable_activation_checkpointing=True \
# max_steps_per_epoch=128