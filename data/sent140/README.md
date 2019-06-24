# Sentiment140 Dataset

## Setup Instructions

Run preprocess.sh with a choice of the following tags:

- ```-s``` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section
- ```--iu``` := number of users, if iid sampling; expressed as a fraction of the total number of users; default is 0.01
- ```--sf``` := fraction of data to sample, written as a decimal; default is 0.1
- ```-k``` := minimum number of samples per user
- ```-t``` := 'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups
- ```--tf``` := fraction of data in training set, written as a decimal; default is 0.9


Instruction used to generate EMNIST in the paper:

```
./preprocess.sh -s niid --sf 1.0 -k 20 -tf 0.8 -t sample
```


(Make sure to delete the rem_user_data, sampled_data, test, and train subfolders in the data directory before re-running preprocess.sh.)

Or you can download the dataset [here](https://drive.google.com/file/d/18bpvQ50qAqKFpbSNNWRUTQSdgooVD453/view?usp=sharing), unzip it and put the `train` and `test` folder under `data`.
