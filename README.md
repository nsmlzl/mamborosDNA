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
* Pyfaidx
* Tensorboard
* Mamba: Build locally and remove `torch_triton_rocm` module; [changes to ROCm 6.0.0 are required](https://github.com/nsmlzl/mamba/blob/cdc6ed054f93c362079aa0b503e7d7bf4ce8f791/README.md?plain=1#L36).
* Triton: Build locally from [official triton repo](https://github.com/triton-lang/triton)

### Modules

```
module load PrgEnv-gnu/8.3.3
module load miniforge3/23.11.0
module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a
```

### Notes

* Removing the Triton cache directory `~/.triton` might resolve some errors.
* Set the environment variable `TRITON_HIP_LLD_PATH` to the path of `ld.lld` (for instance `/opt/rocm/llvm/bin/ld.lld`).
