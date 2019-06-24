#!/usr/bin/env bash

NAME="nist"

cd ../../utils

python3 stats.py --name $NAME

cd ../data/$NAME