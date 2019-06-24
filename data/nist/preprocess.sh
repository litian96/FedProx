#!/usr/bin/env bash

#rm -rf rem_user_data sampled_data test train

# download data and convert to .json format

if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    cd preprocess
    ./data_to_json.sh
    cd ..
fi

NAME="nist" # name of the dataset, equivalent to directory name

cd ../../utils

# ./preprocess.sh -s niid --sf 0.05 -k 64 -t sample
# ./preprocess.sh --name nist -s niid --sf 1.0 -k 0 -t sample
# ./preprocess.sh --name sent140 -s niid --sf 1.0 -k 1 -t sample
./preprocess.sh --name $NAME $@

cd ../data/$NAME
