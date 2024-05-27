# mamborosDNA

WIP code to pre-train the Mamba model on the T2T
([paper](https://doi.org/10.1126/science.abj6987),
[data](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009914755.1/)) and
yeast genomic datasets.

For more information about the ML model, see its [original
preprint](https://doi.org/10.48550/arXiv.2312.00752).


# Running on Frontier

OLCF provides an example to run PyTorch on Frontier ([link](https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html))
and a Frontier user guide ([link](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#software)).

PyTorch Lightning has examples for running multi-node training on SLURM clusters ([link1](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html), [link2](https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html)).

Queue the slurm job *<job.slurm>* with `sbatch --export=NONE <job.slurm>`.


## Requirements
### Python

* PyTorch with ROCm support
* PyTorch Lightning

### Modules

```
module load PrgEnv-gnu/8.3.3
module load miniforge3/23.11.0
module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a
```
