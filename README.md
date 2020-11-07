# easyocr_ros

[![GitHub version](https://badge.fury.io/gh/knorth55%2Feasyocr_ros.svg)](https://badge.fury.io/gh/knorth55%2Feasyocr_ros)
[![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/knorth55/easyocr_ros/CI/master)](https://github.com/knorth55/easyocr_ros/actions)

![sample](./.readme/sample.png)

## Environment

- Ubuntu 16.04 + Kinetic
- Ubuntu 18.04 + Melodic

## Notice

We need `python3.5` or `python3.6` to run this package.

## Setup

### Workspace build

#### Workspace build (Kinetic)

```bash
pip3 install --user opencv-python
source /opt/ros/kinetic/setup.bash
mkdir -p ~/easyocr_ws/src
cd ~/easyocr_ws/src
git clone https://github.com/knorth55/easyocr_ros.git
wstool init
wstool merge easyocr_ros/fc.rosinstall
wstool merge easyocr_ros/fc.rosinstall.kinetic
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/easyocr_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
catkin build
```

#### Workspace build (Melodic)

```bash
sudo apt-get install python3-opencv
source /opt/ros/melodic/setup.bash
mkdir -p ~/easyocr_ws/src
cd ~/easyocr_ws/src
git clone https://github.com/knorth55/easyocr_ros.git
wstool init
wstool merge easyocr_ros/fc.rosinstall
wstool merge easyocr_ros/fc.rosinstall.melodic
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/easyocr_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin build
```
