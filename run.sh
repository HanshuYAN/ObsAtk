CUDA_VISIBLE_DEVICES=0,1,2,3 python main_train_HAT.py --exp_name Gxx25_BSD500_DnCNN_HAT \
                --trainset BSD500 --testset BSD500 --eval_paired CC --network DnCNN-C \
                --batch_size 128 --patch_size 50 --log_interval 100 \
                --highest_noise_level 25 --lowest_noise_level 0 --sigmas4eval 10 15 25 \
                --tr_epochs 50 --milestones 20 40 \
                --alpha 2 \
                --adversary eps5_PGD1