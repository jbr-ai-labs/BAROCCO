#!/usr/bin/env bash
#Pytorch
pip3 install --no-cache-dir http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip3 install --no-cache-dir torchvision

#Ray -- will segfault in .backward with wrong version
#pip uninstall -y ray
#pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/68b11c8251dc41e79dc9dc4b1da31630708a6bd8/ray-0.4.0-cp36-cp36m-manylinux1_x86_64.whl

#Ray
pip3 install ray
pip3 install setproctitle
pip3 install service_identity

#Basics
pip3 install numpy
pip3 install scipy
pip3 install matplotlib


#Tiled map loader
pip3 install pytmx
pip3 install imageio

#jsonpickle
pip3 install jsonpickle
pip3 install opensimplex
pip3 install twisted
