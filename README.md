# flow matching

train
```python 
python ../src/main.py --beta 10.0 --n 6 --folder /data/wanglei/neuralct/firsttry/ --nlayers 4 --nhiddens 512 --backflow
```

inference 
```python 
python ../src/inference.py --beta 10.0 --n 6 --nlayers 4 --nhiddens 512 --backflow  --restore_path /data/wanglei/neuralct/firsttry/n_6_dim_2_beta_10_backflow_nl_4_nh_512/
```
