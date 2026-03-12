The official implementation of Benchmarking Terramind (FM) on **Agriculture Damage Mapping**


### Content Overview
TerraMind for agricultural damage mapping: benchmarking foundation-model transfer across hazards and regions

We frame agricultural damage mapping as a domain-shift problem across:
(1) hazard types
(2) agroecological contexts (cropping types, irrigation regimes, field sizes). 

Our workflow emphasizes commonly available multi-sensor Earth observation data: 
Sentinel-1 SAR offers all-weather sensitivity to surface roughness and moisture changes relevant to inundation, while Sentinel-2 optical reflectance captures vegetation condition and recovery signals. 

---
### Quick Start
To run experiments without pre-conditioners, use:
```bash
torchrun --nproc_per_node=2 damage_mapping/trainer.py -m ++train_loader=small 
```

For scheduled training, use the single SLURM launcher:
```bash
sbatch slurm/terramind.slurm
```

Supported parameters:
- `DATA_SIZE=large` or `DATA_SIZE=small`: choose the dataset size; default is `large`.
- `SLURM_LOG_LEVEL=debug`: print extra environment and command diagnostics.

Examples:
# Run a grid search on large data
```bash
sbatch --export=DATA_SIZE=large_search slurm/terramind.slurm
```

---
### Evaluation

TODO(team): Review `data/input/Images_large/Rejects` S2 tiles and decide whether they should remain excluded or be integrated into the `Train`/`Validation`/`Test` splits as alternate scene candidates.
