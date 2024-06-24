## What does this workflow do?

### Flow structure
The `flow.py` workflow defines a Metaflow `FlowSpec` that dynamically creates the distributed training infrastructure requested in `num_parallel` runtime tasks. Each task is a Kubernetes pod. `@deepspeed` handles setting up inter-process communication, and provides the `current.deepspeed.run` abstraction to wrap the native deepspeed launcher. 
 
### What does this workflow output?
The result of the `train` step is to write 

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