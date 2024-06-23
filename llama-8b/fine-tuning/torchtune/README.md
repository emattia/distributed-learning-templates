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