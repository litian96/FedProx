# MNIST Dataset

First download the raw data [here](https://drive.google.com/file/d/1Vp_gJHw4pPqwMUSgodhFOqUglAQyaOGD/view?usp=sharing), put `mnist-original.mat` under the folder `data/mldata/`.

To generate non-iid data:

```
mkdir test
mkdir train
python generate_niid.py
```

Or you can download the dataset [here](https://drive.google.com/file/d/1cU_LcBAUZvfZWveOMhG4G5Fg9uFXhVdf/view?usp=sharing), unzip it and put the `train` and `test` folder under `data`.

The layout of the folders under `./mnist` should be:

```
| data

----| mldata

---- ----| mnist-original.mat

----| train 

---- ----| train_file_name.json

----| test

---- ----| test_file_name.json

| generate_niid.py
| README.md
```



