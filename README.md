
train
```bash 
python ../src/main.py --folder ../data/ --nlayers 4 --h1size 256 --h2size 64 --keysize 16 --nheads 8 --lr 0.001 --batchsize 100 --ferminet
```

inference 
```bash
python ../src/inference.py  --nlayers 4 --h1size 256 --h2size 64 --keysize 16 --nheads 8  --batchsize 100 --ferminet  --restore_path ../data/traj_N32_rho0.7_T1.0_traj_N32_rho0.7_T1.0_ferminet_l_4_h1_256_h2_64_lr_0.001_fmax_0_bs_100/ --fmax 0
```
