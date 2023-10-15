train
```bash
python ../src/main.py --folder /data/wanglei/he3/rn/ --Nf 5 --nlayers 3 --h1size 32 --h2size 16 --lr 0.001 --fmax 0.0 --batchsize 8192 --X0  /data/zhangqidata/TestHelium3Flow/Helium3FreeFermions_n_14/epoch_000400.pkl --X1 /data/zhangqidata/TestHelium3Flow/Helium3Jastrow_n_14/epoch_004000.pkl --ferminet  --permute  
```

inference 
```bash
 python ../src/inference.py --nlayers 3 --Nf 5 --h1size 32 --h2size 16 --fmax 0.0 --batchsize 8192 --X0  /data/zhangqidata/TestHelium3Flow/Helium3FreeFermions_n_14/epoch_000400.pkl --X1 /data/zhangqidata/TestHelium3Flow/Helium3Jastrow_n_14/epoch_004000.pkl --ferminet --restore_path /data/wanglei/he3/rn/epoch_000400_epoch_004000_permute_ferminet_l_3_Nf_5_h1_32_h2_16_lr_0.001_fmax_0_bs_8192/
```

loss 
```bash
cat /data/wanglei/he3/rn/epoch_000400_epoch_004000_permute_ferminet_l_3_Nf_5_h1_32_h2_16_lr_0.001_fmax_0_bs_8192/loss.txt
```
