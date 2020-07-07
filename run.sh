TRAIN_PATH=data/toy_reverse/train/data.txt
DEV_PATH=data/toy_reverse/dev/data.txt
EXPT_DIR=data/checkpoints/

# Start training
python toy_problem/main.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_DIR