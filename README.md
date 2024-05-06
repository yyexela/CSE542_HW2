# Homework-2

## Setup and Installation
The set up is the same as homework1, skip this if you have finished homework1.
### Install MuJoCo

1. Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
3. Add resources/mjkey.txt in the repo into into `~/.mujoco/mujoco210`.

### Setup environment

To set up the project environment, Use the `environment.yml` file. It contains the necessary dependencies and installation instructions.

    conda env create -f environment.yml
    conda activate cse542a1

### Install LibGLEW

    sudo apt-get install libglew-dev
    sudo apt-get install patchelf
    
### Export paths variables

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    
### Compile mujoco_py (only needs to be done once)
    python -c "import mujoco_py"

## Training 

    python main.py  --task policy_gradient
    python main.py  --task actor_critic
   
## Evaluation 

    python main.py  --task policy_gradient --test --render
    python main.py  --task actor_critic --test --render
    

