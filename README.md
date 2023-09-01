# Training-Free Neural Architecture Search

## Requirements
You can use the following command to reproduce the environment:
```
conda env create -f environment.yml
```

## Pretrained models
We will release the pre-trained models later.

## Architecture performance prediction
To predict the architecture performance on NAS-BNench-101/201, run
```
# NAS-BNench-101
python nasbench101_pred.py --end 1000                           # predict 1000 architectures on CIFAR-10
# NAS-BNench-201
python nasbench201_pred.py --end 1000 --dataset cifar10         # predict 1000 architectures on CIFAR-10
python nasbench201_pred.py --end 1000 --dataset cifar100        # predict 1000 architectures on CIFAR-100
python nasbench201_pred.py --end 1000 --dataset ImageNet16-120  # predict 1000 architectures on ImageNet16-120
python nasbench201_pred.py --dataset cifar10                    # predict all architectures on CIFAR-10
```
`--init_w_type`: weight initialization type `[none, xavier, kaiming, zero, N(0,1)]`; `--batch_size`: batch size for prediction; `--available_measures`: available measures

## Architecture search
To carry out architecture search, run
```
# Search on NAS-BNench-101 space by random search/reinforcement learning/evolutionary algorithm
python search.py --search_space nasbench101 --search_algo random/rl/evolution --dataset cifar10 --measures +act_grad_cor_weighted +synflow +jacob_cor --N 1000
# Search on NAS-BNench-201 space by random search/reinforcement learning/evolutionary algorithm
python search.py --search_space nasbench201 --search_algo random/rl/evolution --dataset cifar10 --measures +act_grad_cor_weighted +synflow +jacob_cor --N 1000
# Search on MobileNetV2 space by evolutionary algorithm
python search.py --search_space MobileNetV2 --search_algo evolution --dataset ImageNet1k --measures +act_grad_cor_weighted +synflow +jacob_cor --N 5000
```

## Architecture evaluation
To evaluate our architecture by training from scratch, run
```
python train_MobileNetV2_on_ImageNet.py --arch EA_our_vote
```
