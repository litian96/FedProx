# Sentiment140 Dataset

## Setup Instructions

You can download the dataset [here](https://drive.google.com/file/d/1pgHf4DUZkGI6q-NLjBzMawX5yn4Y40k0/view?usp=sharing), unzip it and put the `train` and `test` folder under `data`.

If you would like to run on Sent140, you also need to download a pre-trained embedding file using the following commands (this may take 3-5 minutes):

```
cd FedProx_clean_code/flearn/models/sent140
bash get_embs.sh
```

The above dataset is sampled using the following instructions:

Run preprocess.sh with a choice of the following tags:

- ```-s``` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section
- ```--iu``` := number of users, if iid sampling; expressed as a fraction of the total number of users; default is 0.01
- ```--sf``` := fraction of data to sample, written as a decimal; default is 0.1
- ```-k``` := minimum number of samples per user
- ```-t``` := 'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups
- ```--tf``` := fraction of data in training set, written as a decimal; default is 0.9


Instruction used to generate Sent140 in the paper:

```
./preprocess.sh -s niid --sf 0.3 -k 30 -tf 0.8 -t sample
```


(Make sure to delete the rem\_user\_data, sampled\_data, test, and train subfolders in the data directory before re-running preprocess.sh.)


