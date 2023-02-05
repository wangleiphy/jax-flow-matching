
train
```bash 
python ../src/main.py --folder /data/wanglei/lj/npz-temb/ --nlayers 4 --h1size 256 --h2size 64 --keysize 16 --nheads 8 --lr 0.001 --batchsize 100 --dataset ../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz --ferminet  
```

inference 
```bash
python ../src/inference.py --nlayers 4 --h1size 256 --h2size 64 --batchsize 100 --dataset ../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz  --ferminet  --restore_path /data/wanglei/lj/force-unclip-paired-hungarian-L/traj_N32_rho0.7_T1.0_ferminet_l_4_h1_256_h2_64_lr_0.001_fmax_0_bs_1000/ --fmax 0
```
