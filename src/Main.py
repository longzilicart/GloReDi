import argparse

from utils.ct_tools import *
from FreqCDL_trainer import *
from Model.Freq_GloRei import * 
from Sparse_tester import *


def get_parser():
    parser = argparse.ArgumentParser(description='Sparse CT Main')
    # logging interval by iter
    parser.add_argument('--log_interval', type=int, default=400, help='logging interval by iteration')
    # tensorboard
    parser.add_argument('--checkpoint_root', type=str, default='', help='where to save the checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='test', help='detail folder of checkpoint')
    parser.add_argument('--tensorboard_root', type=str, default='', help='root path of tensorboard, project path')
    parser.add_argument('--tensorboard_dir', type=str, required=True, help='detail folder of tensorboard')
    # wandb config
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='Sparse_CT')
    parser.add_argument('--wandb_root', type=str, default='')
    # DDP
    parser.add_argument('--local_rank', type=int, default = -1,
                        help = 'node rank for torch distributed training')
    # data_path
    parser.add_argument('--num_train', type=int, default=40000, help='number of training examples')
    parser.add_argument('--num_val', type=int, default=1000, help='number of validation examples')
    parser.add_argument('--dataset_path', type=str, default = '', help='dataset path')
    # dataset 
    parser.add_argument('--dataset_name', default='deepleision', type=str,
                        help='which dataset, size640,size320,deepleision.etc.')
    parser.add_argument('--dataset_shape', type=int, default = 512,
                        help = 'modify shape in dataset[deepleision dataset]-shape(width, lenght) size of input images, if not resize')
    # ---- hyperparameter args 
    # dataloader
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch_size')    
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='dataloader shuffle, False if test and val')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='dataloader num_workers, 4 is a good choice')
    parser.add_argument('--drop_last', default=False, type=bool,
                        help='dataloader droplast')
    # optimizer
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')    
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='Adam beta1 args')    
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Adam beta2 args')    
    parser.add_argument('--epochs', default=200, type=int,
                        help='training epochs')    
    #step_optimizer
    parser.add_argument('--step_size', default=10, type=int,
                        help='StepLR step_size args')    
    parser.add_argument('--step_gamma', default=0.5, type=float,
                        help='StepLR step_gamma args')

    # checkpath && resume training
    parser.add_argument('--resume', action='store_true',
                        help = 'resume network training or not, load network param')
    parser.add_argument('--resume_opt', action='store_true',
                        help = 'resume optimizer or not, load opt param')
    parser.add_argument('--net_checkpath', default='', type=str,
                        help='network checkpath')
    parser.add_argument('--student_checkpath', default='', type=str,
                        help='teacher network checkpath')
    parser.add_argument('--teacher_checkpath', default='', type=str,
                        help='teacher network checkpath')
    parser.add_argument('--opt_checkpath', default='', type=str,
                        help='optimizer checkpath')
    
    # network hyper args
    parser.add_argument('--trainer_mode', default='train', type=str,
                        help = 'main function - trainer mode, train or test')
    parser.add_argument('--contrast_mode', default='', type=str,
                        help = 'Dudo, transdudo')
    parser.add_argument('--ablation_mode', default='GloRei_E', type=str,
                        help = 'default sparse, cycle: cycle_sparse')
    parser.add_argument('--loss',default='single', type=str,
                        help='default')
    parser.add_argument('--network',default='', type=str,
                        help='network name')
    parser.add_argument('--stn', action='store_true', default=False,
                        help='light weight STN')
    parser.add_argument('--poisson_rate', type=float, default=-1, help='add poisson noise in sinogram')
    parser.add_argument('--gaussian_rate', type=float, default=-1, help='add gaussian noise in sinogram')
    

    # tester args
    parser.add_argument('--tester_save_name',default='default_save', type=str,
                        help='tester_save' )
    parser.add_argument('--tester_save_image',default=False, type=bool,
                        help='' )
    parser.add_argument('--tester_save_path',default='.', type=str,
                        help='tester_save_path' )
    # sparse ct args
    parser.add_argument('--sparse_angle', default=18, type=int,
                        help='sparse angle 18 36 72 is the common setting')
    parser.add_argument('--full_angle', default=720, type=int,
                        help='default 1152')
    # lama args
    parser.add_argument('--lama_ginout',default=0.75, type=float,
                        help='lama global branch setting 0-1, bug if 1')
    parser.add_argument('--e_blocks',default=7, type=int,
                        help='FreqCDL encoder blocks')
    parser.add_argument('--d_blocks',default=2, type=int,
                        help='FreqCDL decoder blocks')
    # cycle ablation when ablation_mode = 'cycle':
    parser.add_argument('--angle_bias_num', default=0, type = int,
                        help = 'augmentation')
    parser.add_argument('--angle_bias_loss', default=False, type = bool,
                        help = 'cycle loss augmentation')
    # loss ablation
    parser.add_argument('--pixel_loss_choice', default='single', choices=['single', 'multi'], help='pixel loss, multi-window loss or single loss')
    parser.add_argument('--frequency_loss_choice', default = None,
                        help='focal is the focal frequency loss')
    parser.add_argument('--sinogram_loss_choice', default = None,
                        help='not implemented yet')
    parser.add_argument('--teacher_loss_choice', default='spatial', 
                        choices=['spatial', 'dct', 'none'],
                        help='calc teacher supervised loss on spatial or dct domain')
    parser.add_argument('--scl_loss_choice', default=None,
                        help='[spatial, dct, None]')
    parser.add_argument('--pretrain_epoch', default=-1, type=int,
                        help='warm up seperately')
    parser.add_argument('--finetune_epoch', default=140, type=int,
                        help='finetune the decoder seperately')
    parser.add_argument('--teacher_dct_mask_ratio', default=0.5, type=float,
                        help='teacher supervised in DCT domain, the ratio')
    # loss weight
    parser.add_argument('--theta_freq', default=1.0, type=float,
                        help='focal frequency loss weight')
    parser.add_argument('--theta_sino', default=0.1, type=float,
                        help='persudo sinogram loss weight')
    parser.add_argument('--theta_tea_sup', default=0.1, type=float,
                        help='teacher supervised loss weight (cosin similarity)')
    parser.add_argument('--theta_scl', default=0.000001, type=float,
                        help='supervised contrastive loss weight')
    parser.add_argument('--theta_cca', default=0.0002, type=float,
                        help='CCA loss weight')
 
    # ---- GloRe config ----
    # network
    parser.add_argument('--down_ginout', default=0, type=float, 
                        help='down_ginout, should be (0, 1)')
    # scl loss
    parser.add_argument('--scl_temperature', default=0.07, type=float, 
                        help='scl_temperature, default: 0.07')
    parser.add_argument('--scl_memory_length', default=300, type=int, 
                        help='scl_memory_length, default 300')
    parser.add_argument('--scl_freq_ratio', default=0.1, type=float, 
                        help='bandpass start default 0.1')
    parser.add_argument('--scl_freq_out_ratio', default=0.6, type=float, 
                        help='bandpass end default 0.5')
    parser.add_argument('--scl_save_mode', choices=['student', 'teacher'],  
                        default='student', type=str, 
                        help='save teacher or student in memory bank')

    # teacher settings
    parser.add_argument('--teacher_angle_scale', default=2, type=int, 
                        help='scale for teacher, opt.sparse_angle * 2. if scale=0, provide teacher_angle')
    parser.add_argument('--teacher_angle', default=720, type=int, 
                        help='if teacher_angle_scale = 0')
    parser.add_argument('--momentum', default=0, type=float, 
                        help='momentum for ema, default 0.9')

    return parser



def sparse_main(opt):

    if opt.ablation_mode in ['GloRei_E']:
        '''framework setting
        GloRei_E: EMA decoder
        '''        
        # [backbone]
        down_ginout = opt.down_ginout 
        global_skip = False
        init_conv_kwargs = {'ratio_gin':down_ginout,'ratio_gout':down_ginout,'enable_lfu':False}
        downsample_conv_kwargs = {'ratio_gin':down_ginout, 'ratio_gout':down_ginout,'enable_lfu':False}
        resnet_conv_kwargs = {'ratio_gin':opt.lama_ginout,'ratio_gout':opt.lama_ginout,'enable_lfu':False}

        # [scl loss]
        scl_learnable = False
        # --- original kwargs ---
        scl_temperature, scl_memory_length = opt.scl_temperature, opt.scl_memory_length
        scl_freq_ratio, scl_freq_out_ratio = [opt.scl_freq_ratio, opt.scl_freq_out_ratio] 
        scl_save_mode = opt.scl_save_mode # ["teacher", "student"]
        momentum = opt.momentum # 0: the same with student encoder
        teacher_angle = (opt.sparse_angle * opt.teacher_angle_scale) if opt.teacher_angle_scale != 0 else opt.teacher_angle
        teacher_angle = int(teacher_angle)
        print(f"teacher angle: {teacher_angle}")

        simulate_kwargs = {'sparse_angle': opt.sparse_angle, 'teacher_angle':teacher_angle, 'dataset_shape': opt.dataset_shape, 'poisson_rate':opt.poisson_rate}
        # define model
        add_out_act = True #


        if opt.ablation_mode in ['GloRei_E']:
            FreqCDL_net = GloRei_EMA
        else:
            raise NotImplementedError('Not supported network')

        print(f"FreqCDL encoder blocks {opt.e_blocks}, decoder blocks {opt.d_blocks}")
        net = FreqCDL_net(1, 1, n_downsampling=2, e_blocks=opt.e_blocks, d_blocks = opt.d_blocks, add_out_act=add_out_act, init_conv_kwargs = init_conv_kwargs, downsample_conv_kwargs = downsample_conv_kwargs, resnet_conv_kwargs = resnet_conv_kwargs,
        scl_memory_length=scl_memory_length, scl_memory_device='cuda', scl_temperature=scl_temperature, scl_mode = opt.scl_loss_choice, scl_low_ratio=scl_freq_ratio, scl_high_ratio=scl_freq_out_ratio,scl_save_mode=scl_save_mode, scl_learnable = scl_learnable, scl_learnable_mode = None, global_skip=global_skip,
        simulate_kwargs = simulate_kwargs, momentum=momentum) 
        
        if opt.ablation_mode == 'GloRei_E':
            print('start training GloRei_EMA')
            sparse_trainer = FreqCDL_Trainer_EMA(opt, sparse_net=net)
        else:
            raise NotImplementedError('Not supported trainer')



    # [train mode]
    if opt.trainer_mode == 'train':
        sparse_trainer.fit()
    elif opt.trainer_mode == 'test':
        sparse_tester = Sparse_tester(opt, net)
        sparse_tester.tester()
    else:
        raise ValueError('opt trainer mode error: must be train or test, not {}'.format(opt.trainer_mode))

    print('finish') # change to log


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    sparse_main(opt)








