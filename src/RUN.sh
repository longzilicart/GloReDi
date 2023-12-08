


# ****************** example for training ****************** 
# config:
    # need to specify basic root
    tensorboard_root=''
    checkpoint_root=''
    dataset_path='' # the base directory for dataset
    wandb_root=''
    # then, tensorboard_dir is the name for tensorboard and wandb
    # checkpoint can be found in (checkpoint_root + checkpoint_dir)
# resume:
    # --resume --resume_opt 
    # --net_checkpath --opt_checkpath
    # resume teacher and student respectively:
    # --teacher_checkpath --student_checkpath

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch \
--master_port 29996 \
--nproc_per_node 8 \
Main.py \
--lr 5e-4 \
--trainer_mode 'train' --sparse_angle 72 --network 'lama' --dataset_shape 256 \
--ablation_mode 'GloRei_E' --e_blocks 7 --d_blocks 2 \
--pretrain_epoch -1 --finetune_epoch 140 \
--teacher_loss_choice 'spatial' --theta_tea_sup 0.002 \
--tensorboard_root $tensorboard_root \
--tensorboard_dir  [O][256][72][1e6]GloRei_E[dis][scl][T2] \
--checkpoint_root $checkpoint_root \
--checkpoint_dir [O][256][72][1e6]GloRei_E[dis][scl][T2] \
--dataset_path $dataset_path \
--batch_size 32 --num_workers 2 --step_size 45 --step_gamma 0.5 --epochs 150 \
--log_interval 100 \
--use_wandb --wandb_project '[sparse_ct]' --wandb_root $wandb_root
# ****************** example for training ****************** 




