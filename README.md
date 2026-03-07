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
```

```

To run experiments **without** pre-conditioners, please use the following slurm command.
```
torchrun --nproc_per_node=2 damage_mapping/trainer.py -m ++train_loader=small 
```

For multi-GPU training, please use the following slurm command (2 GPUs example)
```
sbatch --export=JOB_TYPE=train terramind.slurm
```

---
### Evaluation



