# Shakespeare Dataset

## Setup Instructions
- Run preprocess.sh with a choice of the following tags:

  - ```-s``` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section
  - ```--iu``` := number of users, if i.i.d. sampling; expressed as a fraction of the total number of users; default is 0.01
  - ```--sf``` := fraction of data to sample, written as a decimal; default is 0.1
  - ```-k``` := minimum number of samples per user
  - ```-t``` := 'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups; default is 'sample'
  - ```--tf``` := fraction of data in training set, written as a decimal; default is 0.8
  - ```--raw``` := include users' raw text data in all_data.json

Instruction used to generate EMNIST in the paper:

```
./preprocess.sh -s niid --sf 0.2 -k 0 -tf 0.8 -t sample
```


Make sure to delete the rem_user_data, sampled_data, test, and train subfolders in the data directory before re-running preprocess.sh

Or you can download the dataset [here](https://drive.google.com/file/d/1n46Mftp3_ahRi1Z6jYhEriyLtdRDS1tD/view?usp=sharing), unzip it and put the `train` and `test` folder under `data`.
