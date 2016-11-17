#!/bin/bash
cd build
./cuda_bvh -i ../objs/funyz/robot-dog.obj -o robotdog.bmp -P 12 -a 0.001,0.001,0.01 -f 2.8 -w 1024 -h 1024
