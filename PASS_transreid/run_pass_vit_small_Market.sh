#Single GPU
python train.py --config_file configs/market/vit_small.yml MODEL.PRETRAIN_PATH "/home/mscherbina/Documents/github_repos/PASS-reID/models/pass_transreid_vit_small.pth" OUTPUT_DIR './logs/market/pass_vit_small_full_cat' SOLVER.SEED 42 MODEL.DEVICE_ID '("0")'


