train
```bash
python ../src/main.py --folder /data/wanglei/he3/first_try/ --nlayers 4 --h1size 64 --h2size 32 --lr 0.001 --fmax 0.0 --batchsize 1024 --X0  /data/zhangqidata/TestHelium3Flow/Helium3FreeFermions_n_14/epoch_000400.pkl --X1 /data/zhangqidata/TestHelium3Flow/Helium3Jastrow_n_14/epoch_004000.pkl --ferminet  --permute 
```

inference 
```bash
python ../src/inference.py --nlayers 4 --h1size 64 --h2size 32 --fmax 0.0 --batchsize 8192 --X0  /data/zhangqidata/TestHelium3Flow/Helium3FreeFermions_n_14/epoch_000400.pkl --X1 /data/zhangqidata/TestHelium3Flow/Helium3Jastrow_n_14/epoch_004000.pkl --ferminet --restore_path /data/wanglei/he3/first_try/epoch_000400_epoch_004000_permute_ferminet_l_4_h1_64_h2_32_lr_0.001_fmax_0_bs_1024/
```

loss 
```
cat /data/wanglei/he3/first_try/epoch_000400_epoch_004000_permute_ferminet_l_4_h1_64_h2_32_lr_0.001_fmax_0_bs_1024/loss.txt
```
