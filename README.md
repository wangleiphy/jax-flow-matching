
train
```bash 
python main.py --ferminet --depth 4 --h1size 64 --h2size 16
```

inference 
```bash
python inference.py --ferminet --depth 4 --h1size 64 --h2size 16 --restore_path ../data/n_32_dim_3_lr_0.001_ferminet_d_4_h1_64_h2_16/
```
