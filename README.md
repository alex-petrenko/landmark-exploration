# landmark-exploration
Attempt to develop a new RL algorithm for hard exploration problems

```
Installation (Ubuntu):
1) install VizDoom dependencies: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps

sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip

sudo apt-get install libboost-all-dev

sudo apt install ffmpeg   <-- for gif summaries

2) go to project dir and install python packages:

git clone https://github.com/alex-petrenko/landmark-exploration.git

cd landmark-exploration/

pipenv install

3) install pynput for evaluation scripts (not installed with pipenv because of incompatibility between mac/linux)

pipenv run pip install pynput
sudo apt-get install python3-tk-dbg
```


To train PPO on a simple DOOM environment:
```
pipenv shell # activate virtualenv
python -m algorithms.baselines.ppo.train_ppo --env doom_basic --gpu_mem_fraction=0.3
python -m algorithms.baselines.ppo.train_ppo --env doom_maze --gpu_mem_fraction=0.3
python -m algorithms.baselines.ppo.train_ppo --env doom_maze_very_sparse --gpu_mem_fraction=0.3
```
