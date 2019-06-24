# MNIST Dataset



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

---- ----| raw_data.mat

----| train 

---- ----| train_file_name.json

----| test

---- ----| test_file_name.json

| generate_niid.py
| README.md
```



