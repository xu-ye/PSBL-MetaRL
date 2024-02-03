# PSBL 



## Installation 

```bash
# download source
git clone 
cd PSBL-MetaRL
# make a fresh conda environment with python 3.10
conda create -n PSBL python==3.10
conda activate PSBL
# install core agent
pip install -e .
# PSBL includes built-in support for a number of benchmark environments that can be installed with:
pip install -e .[envs]
```
The default Transformer policy (`nets.traj_encoders.TformerTrajEncoder`) has an option for [FlashAttention 2.0](https://github.com/Dao-AILab/flash-attention). FlashAttention leads to significant speedups on long sequences if your GPU is compatible. We try to install this for you with: `pip install -e .[flash]`, but please refer to the [official installation instructions](https://github.com/Dao-AILab/flash-attention) if you run into issues.



