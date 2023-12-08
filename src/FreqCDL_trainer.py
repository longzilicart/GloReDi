# trainer for distillation network

from Basic_trainer import *
from torch.nn.parameter import Parameter
from Basic_Freq_Module.torch_dct import *
from utils.SCL_loss import SupervisedContrastiveLoss
from itertools import chain


class FreqCDL_Trainer_Basic(Trainer_Basic):
    def __init__(self,opt = None, sparse_net = None, ):
        super(FreqCDL_Trainer_Basic, self).__init__()
        assert opt is not None and sparse_net is not None
        myprint('sparse angle : {}'.format(opt.sparse_angle))
        self.net = sparse_net
        self.opt = opt

        # train val dataset
        if opt.dataset_name == 'deepleision':
            self.train_dataset = Deep_Lesion_Dataset(opt.dataset_path, 'train', dataset_shape = opt.dataset_shape)
            self.val_dataset = Deep_Lesion_Dataset(opt.dataset_path, 'val', dataset_shape = opt.dataset_shape)
        else:
            self.train_dataset = AAPM_Myo_Dataset(opt.dataset_path, 'train', dataset_shape = opt.dataset_shape)
            self.val_dataset = AAPM_Myo_Dataset(opt.dataset_path, 'val', dataset_shape = opt.dataset_shape)

        # checkpath setting
        self.checkpoint_path = os.path.join(opt.checkpoint_root, opt.checkpoint_dir)
        # tensorboard setting
        self.writer = SummaryWriter(os.path.join(opt.tensorboard_root, opt.tensorboard_dir))
        # log interval
        self.itlog_intv = opt.log_interval
        
        # hyperparameter for Trainer
        self.pixel_loss_choice = self.opt.pixel_loss_choice # [multi, single]
        self.frequency_loss_choice = self.opt.frequency_loss_choice # [focal]
        # self.sinogram_loss_choice = self.opt.sinogram_loss_choice # [freq, ]
        self.sinogram_loss_choice = None
        self.teacher_loss_choice = self.opt.teacher_loss_choice # [spatial, dct]
        self.scl_loss_choice = self.opt.scl_loss_choice # [spatial, dct]
        myprint(f"FreqCDL trainer setting: \n pixel_choice:{self.pixel_loss_choice}, frequency_loss_choice:{self.frequency_loss_choice}, sinogram_loss_choice:{self.sinogram_loss_choice}, distillation_loss_choice:{self.teacher_loss_choice}, scl_loss_choice:{self.scl_loss_choice}")
        self.teacher_dct_mask_ratio = self.opt.teacher_dct_mask_ratio        
        self.pretrain_epoch = self.opt.pretrain_epoch 
        self.focal_freq_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)

    def prepare_data(self, miu_ct):
        sparse_ct_miu, dense_ct_miu, gt_ct_miu = self.net.module.sparse_pre_processing_w_multiview(miu_ct, )
        return sparse_ct_miu, dense_ct_miu, gt_ct_miu.detach()


    def dis_loss_fn(self, sparse_latent, dense_latent, mode = 'cosine'):
        assert mode in ['cosine', 'pixel', 'l1', 'l2']
        if mode == 'cosine':
            return self.focal_cosin_loss(sparse_latent, dense_latent, mode = 'cosine_sim', focal = 1)
        elif mode == 'pixel':
            return self.pixel_loss(sparse_latent, dense_latent, mode = 'sml1')
        elif mode == 'l1':
            return self.pixel_loss(sparse_latent, dense_latent, mode = 'l1')
        elif mode == 'l2':
            return self.pixel_loss(sparse_latent, dense_latent, mode = 'l2')            
        else:
            raise NotImplementedError


    # image domain loss calculator
    def spatial_loss_val_fn(self, pred_miu, gt_miu, pixel_loss_mode = 'sml1'):
        ''' spatial_loss [student and teacher]
        1. pixel_loss / multi_window_loss
        2. focal_frequency_loss [not in our implementation]
        3. sinogram_loss (pixel/frequency) [has been deleted]
        '''
        pixel_loss_choice = self.pixel_loss_choice # [multi, single]
        frequency_loss_choice = self.frequency_loss_choice # [focal]
        sinogram_loss_choice = self.sinogram_loss_choice # [freq, ]

        pixel_loss = self.pixel_loss(pred_miu, gt_miu, mode = pixel_loss_mode)
        if frequency_loss_choice == 'focal':
            frequency_loss = self.focal_freq_loss(pred_miu, gt_miu)
        else:
            frequency_loss = torch.tensor(0.0)
        if sinogram_loss_choice == '': 
            sinogram_loss = torch.tensor(0.0)
        else:
            sinogram_loss = torch.tensor(0.0)
        return pixel_loss, frequency_loss, sinogram_loss


    # image domain loss calculator
    def spatial_loss_T_fn(self, pred_miu, gt_miu, pixel_loss_mode = 'sml1'):
        pixel_loss_choice = self.pixel_loss_choice # [multi, single]
        frequency_loss_choice = self.frequency_loss_choice # [focal]
        sinogram_loss_choice = self.sinogram_loss_choice # [freq, ]

        pixel_loss = self.scale_pixel_loss(pred_miu, gt_miu, mode = pixel_loss_mode)
        if frequency_loss_choice == 'focal':
            frequency_loss = self.focal_freq_loss(pred_miu, gt_miu)
        else:
            frequency_loss = torch.tensor(0.0)
        if sinogram_loss_choice == '': # 
            sinogram_loss = torch.tensor(0.0)
        else:
            sinogram_loss = torch.tensor(0.0)
        return pixel_loss, frequency_loss, sinogram_loss


    # image domain loss calculator
    def spatial_loss_S_fn(self, pred_miu, gt_miu, pixel_loss_mode = 'sml1'):
        pixel_loss_choice = self.pixel_loss_choice # [multi, single]
        frequency_loss_choice = self.frequency_loss_choice # [focal]
        sinogram_loss_choice = self.sinogram_loss_choice # [freq, ]

        pixel_loss = self.scale_pixel_loss_student(pred_miu, gt_miu, mode = pixel_loss_mode)
        if frequency_loss_choice == 'focal':
            frequency_loss = self.focal_freq_loss(pred_miu, gt_miu)
        else:
            frequency_loss = torch.tensor(0.0)
        if sinogram_loss_choice == '': 
            sinogram_loss = torch.tensor(0.0)
        else:
            sinogram_loss = torch.tensor(0.0)
        return pixel_loss, frequency_loss, sinogram_loss


    # latent space loss calculator
    def latent_loss_fn(self, student_latent, teacher_latent, dis_mode = 'cosine', channel_ratio = 0.5):
        assert (0 < channel_ratio <= 1)
        if channel_ratio != 1:
            num_channels = student_latent.shape[1]  # Assuming the channel dimension is 1
            n = int(num_channels * channel_ratio)
            # Slice the tensors to consider only the first n channels
            student_latent_sliced = student_latent[:, :n, ...].clone()
            teacher_latent_sliced = teacher_latent[:, :n, ...].clone()
        else:
            student_latent_sliced = student_latent
            teacher_latent_sliced = teacher_latent

        # assert dis_mode in ['cosine', 'pixel']
        teacher_loss_choice = self.teacher_loss_choice # [spatial, dct]
        scl_loss_choice = self.scl_loss_choice # [spatial, dct]
        # teacher_latent.detach()
        if teacher_loss_choice == 'spatial':
            teacher_sup_loss = self.dis_loss_fn(student_latent_sliced, teacher_latent_sliced, mode = dis_mode) 
        elif teacher_loss_choice == 'dct':
            raise NotImplementedError('DCT distillation loss has been deleted')
        else:
            teacher_sup_loss = torch.tensor(0.0)
        
        if scl_loss_choice == 'spatial' or scl_loss_choice == 'dct':
            scl_loss = self.net.module.get_scl_loss(student_latent_sliced, teacher_latent_sliced, scl_mode = scl_loss_choice)
        else:
            scl_loss = torch.tensor(0.0)
        return teacher_sup_loss, scl_loss


        
    def fit(self,):
        opt = self.opt
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend = 'nccl')
        self.cards = torch.distributed.get_world_size()
        myprint(self.cards)

        device = torch.device('cuda', opt.local_rank)
        if self.opt.resume is True:
            self.resume() 
        else:
            try:
                self.weights_init(self.net)
            except Exception as err:
                myprint('init failed: {}'.format(err))
        # network to device
        self.net = self.net.to(device)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                device_ids = [opt.local_rank],
                                                output_device = opt.local_rank,
                                                find_unused_parameters=True)
                                                # i use unsafe operator here
        # start my logger
        wandb_dir = opt.tensorboard_dir
        self.logger = Longzili_Logger(
            log_name = str(wandb_dir),
            project_name = opt.wandb_project,
            config_opt = opt,
            checkpoint_root_path = opt.checkpoint_root,
            tensorboard_root_path = opt.tensorboard_root,
            wandb_root_path = opt.wandb_root,
            use_wandb = True,
            log_interval = opt.log_interval,)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  sampler = train_sampler,
                                  pin_memory = True,) # pin_memory = True
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1,
                                  num_workers=0,
                                  sampler = val_sampler,)
        # init and resume optimizer
        self.init_adam_optimizer(self.net)
        if self.opt.resume_opt is True:
            self.resume_opt()
            myprint('resume{}'.format(self.epoch))
        # start training
        start_epoch = self.epoch
        self.iter = 0
        for self.epoch in range(start_epoch, opt.epochs):
            myprint('start trining epoch:{}'.format(self.epoch))
            self.train_loader.sampler.set_epoch(self.epoch)
            self.train()
            self.val()
            info_dict = {
                'epoch': self.epoch,
                'batch_size': self.opt.batch_size,
                'lr': self.optimizer.state_dict()['param_groups'][0]['lr']}
            self.logger.log_info_dict(info_dict)
            self.step_optimizer.step()
            if self.opt.local_rank == 0:
                self.save_model()
                self.save_opt()

    def train(self,):
        raise NotImplementedError("FreqCDL_Trainer_Basic is not yet implemented")

    @torch.no_grad()
    def val(self,):
        theta_freq = self.opt.theta_freq
        theta_sino = self.opt.theta_sino
        
        # val the model
        self.net.eval()
        pbar = tqdm.tqdm(self.val_loader, ncols = 60)
        for i, data in enumerate(pbar):
            # if i>20: # debug
            #     break
            miu_ct = data
            miu_ct = miu_ct.to('cuda')
            sparse_ct_miu, dense_ct_miu, gt_ct_miu = self.prepare_data(miu_ct)
            # === val step1: val teacher ===
            teacher_output_miu, teacher_latent = self.net([sparse_ct_miu, dense_ct_miu, 'teacher'])
            tea_pixel_loss, tea_freq_loss, tea_sino_loss = self.spatial_loss_val_fn(teacher_output_miu, gt_ct_miu)
            teacher_loss = tea_pixel_loss + tea_freq_loss * theta_freq

            # === val step2: val student ===
            student_output_miu, student_latent = self.net([sparse_ct_miu, dense_ct_miu, 'student'])
            pixel_loss, freq_loss, sino_loss = self.spatial_loss_val_fn(student_output_miu, gt_ct_miu)
            loss = pixel_loss + freq_loss * theta_freq + sino_loss * theta_sino 
            
            # -delete self loss-
            # calculate the accuracy : orindary acc && cycle acc
            rmse, psnr, ssim = self.cal_miu_ct_acc_by_window(student_output_miu, gt_ct_miu)
            iter_log = {
                'loss': loss,
                'teacher_losses': teacher_loss,
                'pixel_losses ': pixel_loss,
                'rmse': rmse,
                'ssim': ssim,
                'psnr': psnr,}
            self.logger.log_scalar_dict(iter_log, log_type='iter', training_stage='val')

        val_img_info = {
            'sparse_ct_miu': sparse_ct_miu,
            'student_output_miu': student_output_miu,
            'dense_ct_miu': dense_ct_miu,
            'teacher_output_miu':teacher_output_miu,
            'gt_ct_miu': gt_ct_miu,}
        self.logger.log_scalar(force=True, log_type='epoch', training_stage = 'val')
        self.logger.log_image_dict(val_img_info, log_type='epoch', force=True, training_stage = 'val_image') 





# =================== FreqCDL_EMA trainer ==================
class FreqCDL_Trainer_EMA(FreqCDL_Trainer_Basic):
    '''EMA trainer'''
    def __init__(self, opt=None, sparse_net=None):
        super().__init__(opt = opt, sparse_net = sparse_net)
        myprint('FreqCDL_EMA Trainer')

    def init_EMA_optimizer(self, net):
        # dont need to optimize decoder2
        self.optimizer = torch.optim.Adam(params=chain(
                        net.module.encoder1.parameters(),
                        net.module.encoder2.parameters(),
                        net.module.decoder.parameters(),
                        ),lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
        self.step_optimizer = StepLR(self.optimizer, step_size = self.opt.step_size, gamma=self.opt.step_gamma)
        myprint('finish initing EMA trainer')

    # Sparse_Freq_Trainer_EMA.fit()
    def fit(self,):
        opt = self.opt
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend = 'nccl')
        device = torch.device('cuda', opt.local_rank)
        self.cards = torch.distributed.get_world_size()
        myprint(self.cards)
        # resume model
        if self.opt.resume is True:
            self.resume() 
        else:
            try:
                self.weights_init(self.net)
            except Exception as err:
                myprint('init failed: {}'.format(err))

        # network to device
        self.net = self.net.to(device)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                device_ids = [opt.local_rank],
                                                output_device = opt.local_rank,
                                                find_unused_parameters=True)

        # start my logger 
        wandb_dir = opt.tensorboard_dir
        self.logger = Longzili_Logger(
            log_name = str(wandb_dir),
            project_name = opt.wandb_project,
            config_opt = opt,
            checkpoint_root_path = opt.checkpoint_root,
            tensorboard_root_path = opt.tensorboard_root,
            wandb_root_path = opt.wandb_root,
            use_wandb = True,
            log_interval = opt.log_interval,)

        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  sampler = train_sampler,
                                  pin_memory = True,) # pin_memory = True
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1,
                                  num_workers=0,
                                  sampler = val_sampler,)

        # init and resume optimizer
        self.init_EMA_optimizer(self.net)
        if self.opt.resume_opt is True:
            self.resume_opt()
            myprint('resume{}'.format(self.epoch))

        # start training
        start_epoch = self.epoch
        self.iter = 0
        for self.epoch in range(start_epoch, opt.epochs):
            myprint('start trining epoch:{}'.format(self.epoch))
            self.train_loader.sampler.set_epoch(self.epoch)
            self.train()
            self.val()
            info_dict = {
                'epoch': self.epoch,
                'batch_size': self.opt.batch_size,
                'lr': self.optimizer.state_dict()['param_groups'][0]['lr']}
            self.logger.log_info_dict(info_dict)
            self.step_optimizer.step()
            if self.opt.local_rank == 0:
                self.save_model()
                self.save_opt()


    def train(self, ):
        _pretrain_epoch = self.opt.pretrain_epoch
        _finetune_epoch = self.opt.finetune_epoch
        if self.epoch < _pretrain_epoch:
            _stage = 'pretrain'
            myprint('pretrain, Training separately')
        else: 
            _stage = 'FreqCDL'
            myprint('FreqCDL: distillation achitecture')
        if self.epoch > _finetune_epoch:
            myprint('finetune the network')
            _stage = 'finetune'

        theta_freq = self.opt.theta_freq
        theta_sino = self.opt.theta_sino
        theta_tea_sup = self.opt.theta_tea_sup
        theta_scl = self.opt.theta_scl
        
        pbar = tqdm.tqdm(self.train_loader, ncols = 60)
        self.net.train()
        for i, data in enumerate(pbar):
            # if i>52: # debug
            #     break
            miu_ct = data
            miu_ct = miu_ct.to('cuda')
            sparse_ct_miu, dense_ct_miu, gt_ct_miu = self.prepare_data(miu_ct)
            # === step1: train teacher network === #
            if _stage != 'finetune':
                self.net.module.update_EMA_decoder_param()

                teacher_output_miu, _ = self.net([sparse_ct_miu, dense_ct_miu, 'teacher'])
                tea_pixel_loss, tea_freq_loss, tea_sino_loss = self.spatial_loss_T_fn(teacher_output_miu, gt_ct_miu) 
                teacher_loss = tea_pixel_loss + tea_freq_loss * theta_freq + tea_sino_loss * theta_sino 
                self.optimizer.zero_grad()
                # teacher_loss.backward()
                self.reduce_loss(teacher_loss).backward()
                self.optimizer.step()

                with torch.no_grad():
                    teacher_latent = self.net([sparse_ct_miu, dense_ct_miu, 'teacher_sup'])
                    
                # === step2: train student network === #
                student_output_miu, student_latent = self.net([sparse_ct_miu, dense_ct_miu, 'student'])
                pixel_loss, freq_loss, sino_loss = self.spatial_loss_S_fn(student_output_miu, gt_ct_miu)
                if _stage == 'pretrain':
                    teacher_sup_loss, scl_loss = torch.tensor(0.0), torch.tensor(0.0)
                else:
                    teacher_sup_loss, scl_loss = self.latent_loss_fn(student_latent, teacher_latent, dis_mode = 'cosine')
                loss = pixel_loss + freq_loss * theta_freq + sino_loss * theta_sino + teacher_sup_loss * theta_tea_sup + scl_loss * theta_scl
                self.optimizer.zero_grad()
                self.reduce_loss(loss).backward()
                self.optimizer.step()

            else: # _stage == 'finetune
                with torch.no_grad():
                    teacher_output_miu, _ = self.net([sparse_ct_miu, dense_ct_miu, 'teacher'])
                    teacher_loss = 0
                student_output_miu, student_latent = self.net([sparse_ct_miu, dense_ct_miu, 'finetune'])
                pixel_loss, freq_loss, sino_loss = self.spatial_loss_fn(student_output_miu, gt_ct_miu)
                teacher_sup_loss, scl_loss = torch.tensor(0.0), torch.tensor(0.0)
                loss = pixel_loss + freq_loss * theta_freq + sino_loss * theta_sino + teacher_sup_loss * theta_tea_sup + scl_loss * theta_scl
                self.optimizer.zero_grad()
                # loss.backward()
                self.reduce_loss(loss).backward()
                self.optimizer.step()

            rmse, psnr, ssim = self.cal_miu_ct_acc_by_window(student_output_miu, gt_ct_miu)
            # my logger
            self.logger.tick() # iter tick
            iter_log = {
                'loss': loss,
                'teacher_losses': teacher_loss,
                'pixel_losses ': pixel_loss,
                'teacher_sup_loss': teacher_sup_loss * theta_tea_sup,
                'scl_losses': scl_loss * theta_scl,
                'rmse': rmse, 'ssim': ssim, 'psnr': psnr,}
            self.logger.log_scalar_dict(iter_log, log_type = 'iter', training_stage='train')
            

        # epoch log
        self.logger.log_scalar(force=True, log_type='epoch', training_stage = 'train')
        img_info = {
            'sparse_ct_miu': sparse_ct_miu,
            'student_output_miu': student_output_miu,
            'dense_ct_miu': dense_ct_miu,
            'teacher_output_miu': teacher_output_miu,
            'gt_ct_miu': gt_ct_miu,}
        self.logger.log_image_dict(img_info, log_type='epoch', force=True, training_stage = 'train_image')    







if __name__ == '__main__':
    pass





