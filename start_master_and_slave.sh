#!/bin/bash
start-master.sh
start-slave.sh spark://spark:7077

curl https://bootstrap.pypa.io/get-pip.py > get-pip.py
python get-pip.py
pip install numpy