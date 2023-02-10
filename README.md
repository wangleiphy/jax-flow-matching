
train
```bash 
python ../src/main.py --folder /data/wanglei/lj/force-unclip-paired-hungarian-L-wca-lj/ --nlayers 4 --h1size 256 --h2size 64 --keysize 16 --nheads 8 --lr 0.001 --fmax 0.0 --batchsize 1000 --X0 ../data/LJTraj_WCA/liquid/traj_N32_rho0.7_T1.0.npz --X1 ../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz --ferminet  --permute  
```

inference 
```bash
python ../src/inference.py --nlayers 4 --h1size 256 --h2size 64 --fmax 0.0 --batchsize 1000 --X0 ../data/LJTraj_WCA/liquid/traj_N32_rho0.7_T1.0.npz --X1 ../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz --ferminet  --restore_path /data/wanglei/lj/force-unclip-paired-hungarian-L-wca-lj/traj_N32_rho0.7_T1.0_traj_N32_rho0.7_T1.0_permute_ferminet_l_4_h1_256_h2_64_lr_0.001_fmax_0_bs_1000/
```

loss and free energy bound
```
cat /data/wanglei/lj/force-unclip-paired-hungarian-L-wca-lj/traj_N32_rho0.7_T1.0_traj_N32_rho0.7_T1.0_permute_ferminet_l_4_h1_256_h2_64_lr_0.001_fmax_0_bs_1000/loss.txt
cat /data/wanglei/lj/force-unclip-paired-hungarian-L-wca-lj/traj_N32_rho0.7_T1.0_traj_N32_rho0.7_T1.0_permute_ferminet_l_4_h1_256_h2_64_lr_0.001_fmax_0_bs_1000/fe.txt
```
