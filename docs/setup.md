# Setup 

This repository makes use of several external libraries. 
We highly recommend installing them within a virtual environment such as Anaconda. 

The script below will help you set up the environment; the `--yes` flag allows conda to install
without requesting your input for each package.

```bash 
conda create --name nvp python=3.6 --yes 
conda activate nvp

conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch --yes

# Other packages for plotting
conda install tensorboard matplotlib --yes

# multiprocessing / plotting
pip install ray joblib

# Other dependances
pip install scipy opencv-python seaborn
```


