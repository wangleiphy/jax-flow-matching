
train
```bash 
python ../src/main.py --folder /data/wanglei/lj/npz-temb/ --nlayers 4 --h1size 256 --h2size 64 --keysize 16 --nheads 8 --lr 0.001 --batchsize 100 --dataset ../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz --ferminet  
```

inference 
```bash
python ../src/inference.py --nlayers 4 --h1size 256 --h2size 64 --batchsize 100 --dataset ../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz  --ferminet  --restore_path /data/wanglei/lj/npz-temb/traj_N32_rho0.7_T1.0_ferminet_l_4_h1_256_h2_64lr_0.001_bs_100/
```
