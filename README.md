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

Queue the slurm job *<job.slurm>* with `sbatch --export=NONE <job.slurm>`.


## Requirements
### Python

* PyTorch with ROCm support
* mpi4py
