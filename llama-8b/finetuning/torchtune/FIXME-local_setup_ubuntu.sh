export PYTHON_VERSION=3.12
export ENV_NAME=torchtune
export MAMBA_ROOT_PREFIX=~/micromamba

# Ensure Micromamba is in the PATH
export PATH=~/bin:$PATH

function check_micromamba_installed() {
    if [[ -x "$(command -v micromamba)" ]]; then
        return 0
    else
        return 1
    fi
}

function check_env_exists() {
    local env_name=$1
    if micromamba env list | grep -q "$env_name"; then
        echo "Environment '$env_name' already exists."
        return 0
    else
        return 1
    fi
}

if ! check_micromamba_installed; then
    cd ~ && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate
    ./bin/micromamba shell init -s bash -p ~/micromamba
    source ~/.bashrc
else
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate
    echo "Micromamba is already installed."
fi

if ! check_env_exists $ENV_NAME; then
    micromamba create -n $ENV_NAME python=$PYTHON_VERSION pip -c conda-forge -y
    micromamba activate $ENV_NAME
    git clone https://github.com/pytorch/torchtune.git
    cd torchtune && pip install . && cd .. && rm -rf torchtune
else
    echo "Activate:"
    echo "    micromamba activate torchtune"
    micromamba activate $ENV_NAME
fi

# Set environment variables from the .env file
cd ~/distributed-learning-templates/llama-8b/finetuning/torchtune
set -a
source .env
set +a
