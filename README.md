# Federated Optimization in Heterogeneous Networks

This repository contains the code and experiments for the paper:

> [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
>
> [MLSys 2020](https://mlsys.org/)

Federated Learning is a distributed learning paradigm with two key challenges that differentiate it from traditional distributed optimization: (1) significant variability in terms of the systems characteristics on each device in the network (systems heterogeneity), and (2) non-identically distributed data across the network (statistical heterogeneity). In this work, we introduce a framework, FedProx, to tackle heterogeneity in federated networks, both theoretically and empirically. 

This repository contains a set of detailed empirical evaluation across a suite of federated datasets. We show that FedProx allows for more robust convergence than FedAvg. In particular, in highly heterogeneous settings, FedProx demonstrates significantly more stable and accurate convergence behavior relative to FedAvg—improving absolute test accuracy by 22% on average.

## General Guidelines

Note that if you would like to use FedProx as a baseline and run our code:

* If you are using different datasets, then at least the learning rates and the mu parameter need to be tuned based on your metric. You might want to tune mu from {0.001, 0.01, 0.1, 0.5, 1}. There are no default mu values that would work for all settings.

* If you are using the same datasets as those used here, then need to use the same learning rates and mu reported in our [paper](https://arxiv.org/abs/1812.06127).


## Preparation

### Dataset generation

We **already provide four synthetic datasets** that are used in the paper under corresponding folders. For all datasets, see the `README` files in separate `data/$dataset` folders for instructions on preprocessing and/or sampling data.

The statistics of real federated datasets are summarized as follows.

<center>

| Dataset       | Devices         | Samples|Samples/device <br> mean (stdev) |
| ------------- |-------------| -----| ---|
| MNIST      | 1,000 | 69,035 | 69 (106)| 
| FEMNIST     | 200      |   18,345 | 92 (159)|
| Shakespeare | 143    |    517,106 | 3,616 (6,808)|
| Sent140| 772      |    40,783 | 53 (32)|

</center>

### Downloading dependencies

```
pip3 install -r requirements.txt  
```

## Run on synthetic federated data 
(1) You don't need a GPU to run the synthetic data experiments:

```
export CUDA_VISIBLE_DEVICES=
```

(2) Run the instructions as follows, and the log files will be automatically stored for drawing figures later.


```
bash run_fedavg.sh synthetic_iid 0 | tee log_synthetic/synthetic_iid_client10_epoch20_mu0
bash run_fedprox.sh synthetic_iid 0 1 | tee log_synthetic/synthetic_iid_client10_epoch20_mu1
bash run_fedavg.sh synthetic_0_0 0 | tee log_synthetic/synthetic_0_0_client10_epoch20_mu0
bash run_fedprox.sh synthetic_0_0 0 1 | tee log_synthetic/synthetic_0_0_client10_epoch20_mu1
bash run_fedavg.sh synthetic_0.5_0.5 0 | tee log_synthetic/synthetic_0.5_0.5_client10_epoch20_mu0
bash run_fedprox.sh synthetic_0.5_0.5 0 1 | tee log_synthetic/synthetic_0.5_0.5_client10_epoch20_mu1
bash run_fedavg.sh synthetic_1_1 0 | tee log_synthetic/synthetic_1_1_client10_epoch20_mu0
bash run_fedprox.sh synthetic_1_1 0 1 | tee log_synthetic/synthetic_1_1_client10_epoch20_mu1
```

(3) Draw figures to reproduce results on synthetic data

```
python plot_fig2.py loss     # training loss
python plot_fig2.py accuracy # testing accuracy
python plot_fig2.py dissim   # dissimilarity metric

```


The training loss, testing accuracy, and dissimilarity metric figures are saved as `loss.pdf`, `accuracy.pdf` and `dissim.pdf` respectively, under the current folder where you call `plot_fig2.py`. You can check that these figures reproduce the results in Figure 2 in the paper. Make sure to use the default hyper-parameters in `run_fedavg.sh/run_fedprox.sh` for synthetic data. 

For example, the training loss for synthetic datasets would look like this:


![](https://user-images.githubusercontent.com/14993256/52826183-dbf06e80-308d-11e9-9e12-508c3c0a26bf.png)


## Run on real federated datasets
(1) Specify a GPU id if needed:

```
export CUDA_VISIBLE_DEVICES=available_gpu_id
```
Otherwise just run to CPUs [might be slow if testing on Neural Network models]:

```
export CUDA_VISIBLE_DEVICES=
```

(2) Run on one dataset. First, modify the `run_fedavg.sh` and `run_fedprox.sh` scripts, specify the corresponding model of that dataset (choose from `flearn/models/$DATASET/$MODEL.py` and use `$MODEL` as the model name), specify a log file name, and configure all other parameters such as learning rate (see all hyper-parameters values in the appendix of the paper).


For example, for all the synthetic data:

`fedavg.sh`:

```
python3  -u main.py --dataset=$1 --optimizer='fedavg'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --drop_percent=$2 \
            --model='mclr' 
```

`fedprox.sh`:

```
python3  -u main.py --dataset=$1 --optimizer='fedprox'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --drop_percent=$2 \
            --model='mclr' \
            --mu=$3
```


Then run:

```
mkdir synthetic_1_1
bash run_fedavg.sh synthetic_1_1 0 | tee synthetic_1_1/fedavg_drop0
bash run_fedprox.sh synthetic_1_1 0 0 | tee synthetic_1_1/fedprox_drop0_mu0
bash run_fedprox.sh synthetic_1_1 0 1 | tee synthetic_1_1/fedprox_drop0_mu1

bash run_fedavg.sh synthetic_1_1 0.5 | tee synthetic_1_1/fedavg_drop0.5
bash run_fedprox.sh synthetic_1_1 0.5 0 | tee synthetic_1_1/fedprox_drop0.5_mu0
bash run_fedprox.sh synthetic_1_1 0.5 1 | tee synthetic_1_1/fedprox_drop0.5_mu1

bash run_fedavg.sh synthetic_1_1 0.9 | tee synthetic_1_1/fedavg_drop0.9
bash run_fedprox.sh synthetic_1_1 0.9 0 | tee synthetic_1_1/fedprox_drop0.9_mu0
bash run_fedprox.sh synthetic_1_1 0.9 1 | tee synthetic_1_1/fedprox_drop0.9_mu1
```

And the test accuracy, training loss, and dissimilarity numbers will be saved in the log files.

(3) After you collect logs for all the 5 datasets in Figure 1 (synthetic, mnist, femnist, shakespeare, sent140) (the log directories should be `[synthetic_1_1, mnist, femnist, shakespeare, sent140]`), run:

```
python plot_final_e20.py loss
```
to reproduce results in Figure 1 (the generated figure is called `loss_full.pdf`).

*Note: If you only want to quickly verify the results on the first synthetic dataset, you can modify the `plot_final_e20.py` script by changing `range(5)` in Line 54 to `range(1)`, and run `python plot_final_e20.py loss`*.

*Note: It might take a much longer time to run on real datasets than synthetic data because real federated datasets are larger and some of the models are deep neural networks.*


## References
See our [FedProx](https://arxiv.org/abs/1812.06127)  paper for more details as well as all references.
