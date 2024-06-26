## Setup

### Ubuntu
```bash
bash ./local_setup_ubuntu.sh
```

### Download weights

```bash
tune download meta-llama/Meta-Llama-3-8B \
    --output-dir /tmp/Meta-Llama-3-8B \
    --hf-token $HF_TOKEN
```

## Interactive

### Run the workflow from a dev notebook
Open a notebook server:
```bash
jupyter lab
```

Run [`dev.ipynb`](./dev.ipynb).

### Run the workflow from CLI
```bash
python flow.py run
```

## Automated

### Deploy the workflow from CLI
```bash
python flow.py argo-workflows create
```

### Manually trigger from CLI
```bash
python flow.py argo-workflows trigger
```

### No-code trigger from Outerbounds console
...