
train
```bash 
python ../src/main.py --folder /data/wanglei/lj/force/ --nlayers 4 --h1size 256 --h2size 64 --keysize 16 --nheads 8 --lr 0.001 --batchsize 100 --dataset ../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz --ferminet 
```

inference 
```bash
python ../src/inference.py --nlayers 4 --h1size 256 --h2size 64 --batchsize 100 --dataset ../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz --ferminet  --restore_path /data/wanglei/lj/force/n_32_dim_3_lr_0.001_ferminet_l_4_h1_256_h2_64/
```
