# MDAN x DECISION

## Training

首先，预训练source-free源域模型。

```
python train_source_free.py --dset digit --s 0 --max_epoch 40 --trte val --gpu_id 0 --output ckps/source_free/
```

然后进行训练

```
python mix_value.py --dset digit --t 1 --gpu_id 0
```







