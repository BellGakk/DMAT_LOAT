python train-wa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN34-10_cifar10s_lr0p2_MART5_epoch100_bs512_fraction0p7_ls0p1_reg_mse' \
    --data cifar10s \
    --batch-size 128 \
    --batch-size-validation 128 \
    --model wrn-34-10\
    --num-adv-epochs 100 \
    --lr 0.2 \
    --beta 5.0 \
    --mart \
    --more_reg mse \
    --unsup-fraction 0.7 \
    --aux-data-filename '/home/xiangyu/mnt/cifar10/1m.npz' \
    --ls 0.1