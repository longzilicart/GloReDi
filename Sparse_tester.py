# from sklearn.cluster import FeatureAgglomeration
from Basic_trainer import *

from torchvision.utils import make_grid as make_grid
from torchvision.utils import save_image
import pandas as pd
from collections import defaultdict



class Sparse_tester(Trainer_Basic):
    '''Sparse_CT tester
    args:
        opt: the trainer opt
        cttool: cttool init by traner
        save_fig: save pic or not
    '''
    def __init__(self, opt, net, 
                test_win = None,
                data_range = 1,
                **kwargs):
        super(Sparse_tester, self).__init__()
        self.opt = opt
        self.net = net
        self.sparse_angle = self.opt.sparse_angle
        self.cttool = CT_Preprocessing()
        if test_win is not None:
            assert isinstance(test_win, list)
        else:
            test_win = [(3000,500),(500,50),(800,-600)]
        self.test_win = test_win
        self.save_fig = self.opt.tester_save_image

        # 彩色绘图
        if 'rgb_mode' in kwargs:
            self.rgb_mode = kwargs['rgb_mode']
            if 'rgb_dict' in kwargs:
                self.rgb_dict = kwargs['rgb_dict']
        else:
            self.rgb_mode = False
        
        self.data_range = data_range
        
        # tester data
        self.matrix_dict = defaultdict(dict)
        self.save_dir = os.path.join(self.opt.tester_save_path, self.opt.tester_save_name) 
        # 【TODO add tester_save_path and tester_save_name in args】
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.saved_slice = 0
        self.seed_torch(seed=1)

    def seed_torch(self, seed=1):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def prepare_dataset(self, ):
        print('here')
        opt = self.opt
        print(opt.dataset_name)
        if opt.dataset_name == 'deepleision':
            print('Evaluating on deeplesion Dataset')
            self.test_dataset = Deep_Lesion_Dataset(opt.dataset_path, 'test', dataset_shape = opt.dataset_shape)
        elif opt.dataset_name == 'AAPM':
            print('Evaluating on AAPM Myo Dataset')
            self.test_dataset = AAPM_Myo_Dataset(opt.dataset_path, 'test', dataset_shape = opt.dataset_shape)
        else:
            raise NotImplemented(f'dataset {opt.dataset_name} not implemented')
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=1,)

    # 【here】所有方法的forward
    def test_model(self, data):
        if 'Glo' in self.opt.network:
            sparse_ct_miu, dense_ct_miu, gt_ct_miu = self.net.sparse_pre_processing_w_multiview(data, )
            test_dict = {'sps': sparse_ct_miu}
            student_output_miu, student_latent = self.net([sparse_ct_miu, dense_ct_miu, 'student'])
            test_dict = {
                'out': student_output_miu,
            }
            return gt_ct_miu, test_dict
        else:
            raise NotImplementedError(f'network {self.opt.network} not implemented')

    def tester(self,):
        # a simple tester
        if self.opt.network != 'fbp':
            self.load_model(strict=True)
        self.net = self.net.cuda()
        self.prepare_dataset()
        self.net = self.net.eval()
        pbar = tqdm.tqdm(self.test_loader, ncols = 60)
        with torch.no_grad():
            for i, data in enumerate(pbar):
                data = data.cuda()
                gt_ct_miu, test_dict = self.test_model(data)
                self.batch_sparse_tester(gt_ct_miu, **test_dict)
        stat_matrix_dic = self.stat_matrix_dic(self.matrix_dict)
        stat_matrix_df = self.get_df_from_dict(stat_matrix_dic)
        stat_matrix_df.to_csv(os.path.join(self.save_dir, self.opt.network + str(self.opt.sparse_angle) +'_dic.csv'),)
        myprint(stat_matrix_df)

    def batch_sparse_tester(self, gt_ct=None, **kwargs):
        # gt_ct is the gt, use the **first element** in kwargs to calc matrix
        # if save_fig is True, save other pic to window_ct
        assert gt_ct is not None and len(kwargs)>0
        key_list = [k for k in kwargs.keys()]
        batch, _, _, _ = kwargs[key_list[0]].size()
        if batch == 1:
            self.sparse_tester(gt_ct, **kwargs)
        else:
            for b in range(batch):
                single_kwargs = {}
                for key in key_list:
                    single_kwargs[key] = kwargs[key][b:b+1]
                self.sparse_tester(gt_ct[b:b+1], single_kwargs)

    def sparse_tester(self, gt_ct=None, **kwargs):
        for (width, center) in self.test_win:
            calc_flag = -1
            for key, value in kwargs.items():
                if calc_flag < 0:
                    normhu_gtct = self.get_window_norm(gt_ct)
                    normhu_value = self.get_window_norm(value)
                    win_gtct = self.get_window_x_from_norm(normhu_gtct,width=width, center=center)
                    win_value = self.get_window_x_from_norm(normhu_value,width=width, center=center)
                    rmse, psnr, ssim = compute_measure(win_value, win_gtct, self.data_range)
                    try:
                        self.matrix_dict[f'{width}_{center}']['psnr'].append(psnr)
                        self.matrix_dict[f'{width}_{center}']['ssim'].append(ssim)
                        self.matrix_dict[f'{width}_{center}']['rmse'].append(rmse)
                    except:
                        self.matrix_dict[f'{width}_{center}']['psnr'] = []
                        self.matrix_dict[f'{width}_{center}']['ssim'] = []
                        self.matrix_dict[f'{width}_{center}']['rmse'] = []
                        self.matrix_dict[f'{width}_{center}']['psnr'].append(psnr)
                        self.matrix_dict[f'{width}_{center}']['ssim'].append(ssim)
                        self.matrix_dict[f'{width}_{center}']['rmse'].append(rmse)
                    calc_flag += 1
                # save image
                normhu_value = self.get_window_norm(value)
                win_value = self.get_window_x_from_norm(normhu_value,width=width, center=center)
                # if self.save_fig and (width == 2000 and center == 0): 
                if self.save_fig: 
                    psnr_str = '%.2f'%(psnr)
                    ssim_str = '%.2f'%(100*ssim)
                    self.save(win_value, f'{key}[{self.opt.sparse_angle}]_[{width},{center}]_p{psnr_str}_s{ssim_str}')

            if self.opt.network == 'fbp' and self.save_fig:
                normhu_gtct = self.get_window_norm(gt_ct)
                win_gtct = self.get_window_x_from_norm(normhu_gtct,width=width, center=center)
                self.save(win_gtct, f'{"gt"}_[{width},{center}]_p100_s100')
                normhu_spct = self.get_window_norm(value)
                win_spct = self.get_window_x_from_norm(normhu_spct,width=width, center=center)
                self.save(win_spct, f'{"sp"}_[{width},{center}]_p100_s100')
        self.saved_slice += 1

    def save(self, value, name):
        save_dir = self.save_dir
        len_num = int(np.log10(len(self.test_dataset))) + 1
        slice_str = str(self.saved_slice).rjust(len_num, '0')
        net_str = self.opt.network.lower()
        fullname = f'{slice_str}_{name}_{net_str}.png'
        save_path = os.path.join(save_dir, fullname)
        save_image(value, save_path, normalize=False)

    # analysis tool
    def get_mean_from_dict(self, ):
        pass

    def get_df_from_dict(self, double_dict):
        if double_dict is None:
            double_dict = self.matrix_dict
        index_key, columns_key = self.get_columns_index_key(double_dict) 

        # init a dataframe
        df_matrix = pd.DataFrame(columns=columns_key, index = index_key)
        for i in columns_key:
            for j in index_key:
                df_matrix.loc[j,i] = double_dict[j][i] 
        return df_matrix

    def get_columns_index_key(self, double_dic,):
        columns_key_t = double_dic.keys()
        columns_key = []
        for t in columns_key_t:
            columns_key.append(t) #.cpu().numpy().item() 
        columns_key = self.num_str_sort(columns_key)
        index_key = double_dic[columns_key[0]].keys()
        return columns_key, index_key

    # ---- cttools ----
    def get_window_x_from_norm(self, x, width, center):
        HUtensor = self.cttool.back_window_transform(x,width=3000,center=500)
        window_tensor = self.cttool.window_transform(HUtensor,width=width,center=center)
        return window_tensor

    def get_window_norm(self, x):
        return self.cttool.window_transform(self.cttool.miu2HU(x))
    
    def get_window_normHU(self, x):
        return self.cttool.window_transform(x)



    @staticmethod
    def stat_matrix_dic(matrix_dic):
        stat_matrix_dict = defaultdict(dict)
        for columns_key in matrix_dic.keys():
            for row_key in matrix_dic[columns_key]:
                mean_value = np.mean(matrix_dic[columns_key][row_key])
                std_value = np.std(matrix_dic[columns_key][row_key])
                min_value = np.min(matrix_dic[columns_key][row_key])
                max_value = np.max(matrix_dic[columns_key][row_key])
                stat_matrix_dict[columns_key][f'avg_{row_key}'] = mean_value
                stat_matrix_dict[columns_key][f'std_{row_key}'] = std_value
                stat_matrix_dict[columns_key][f'min_{row_key}'] = min_value
                stat_matrix_dict[columns_key][f'max_{row_key}'] = max_value
        return stat_matrix_dict
    
    @staticmethod
    def num_str_sort(num_str_list):
        num_list = []
        str_list = []
        for i in num_str_list:
            if type(i) != type('str'):
                num_list.append(i)
            else:
                str_list.append(i)
        num_list.sort()
        str_list.sort()
        num_str_list = []
        num_str_list.extend(num_list) 
        num_str_list.extend(str_list) 
        return num_str_list

        

    # ----------- no use ----------
    def fit(self,):
        pass
    def train(self,):
        pass
    def val(self,):
        pass









