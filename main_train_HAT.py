from _preamble import *

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--SEED', default=0, type=int)
parser.add_argument('--root', default='./experiments')
# Exp
parser.add_argument('--exp_name', default='test')
parser.add_argument('--trainset', default='Train400')
parser.add_argument('--testset', default='Set68')
# parser.add_argument('--eval_paired', default=None, choices=['SIDD', 'Fluorescence', None])
parser.add_argument('--eval_paired', default=None)
parser.add_argument('--log_interval', default=150, type=int)

parser.add_argument('--highest_noise_level', default=55, type=int)
parser.add_argument('--lowest_noise_level', default=0, type=int)
parser.add_argument('--sigmas4eval', nargs='+', default=[15,25], type=int)

parser.add_argument('--network', type=str, default='DnCNN-B')
parser.add_argument('--patch_size', default=50, type=int)
parser.add_argument('--batch_size', default=128, type=int)


parser.add_argument('--tr_epochs', default=75, type=int)
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
parser.add_argument('--milestones', nargs='+', default=[30, 50, 70], type=int)
parser.add_argument('--gamma', default=0.2, type=str)


parser.add_argument('--adversary', type=str, default='eps5_PGD1')
parser.add_argument('--alpha', type=float, default=1)

# Resume
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint', type=str, default='./experiments/GxxTrain400_DnCNNb_NT/nets/ckp_latest.pt')
parser.add_argument('--net_only', default=True, type=lambda x: bool(int(x)))
args, _ = parser.parse_known_args()

np.random.seed(args.SEED)
torch.manual_seed(args.SEED)


def add_gaus_noise_to_img(x, sigma=None):
    # x, float 0-1
    x = x.detach().clone()
    assert len(x.shape)==4
    if sigma is not None:
        noise = torch.randn(x.shape) * sigma.view(-1,1,1,1)
        x_noise = x + noise.to(x)
        return torch.clamp(x_noise, min=0, max=1)
    else:
        return x

from datasets.image_processing import *
from models.BaseModel import BaseModelDNN
from models.nets.DnCNN import DnCNN
from utils import get_logger

if args.network == 'DnCNN-B':
    def Network():
        print('using DnCNN-B')
        return DnCNN(depth=20, img_channels=1)
elif args.network == 'DnCNN-S':
    def Network():
        print('using DnCNN-S')
        return DnCNN(depth=17, img_channels=1)
elif args.network == 'DnCNN-C':
    def Network():
        print('using DnCNN-C')
        return DnCNN(depth=20, img_channels=3)
else:
    assert False

class Denoiser(BaseModelDNN):
    def __init__(self, args=None, device='cuda', is_train=False) -> None:
        super().__init__()
        self.net = Network().to(device)
        self.is_train = is_train
        self.device = device
        self.GPU_IDs = list(range(torch.cuda.device_count()))
        self.highest_noise_level = args.highest_noise_level
        self.lowest_noise_level = args.lowest_noise_level
        if len(self.GPU_IDs) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=self.GPU_IDs)
        if is_train:
            self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
            self.criterion = nn.MSELoss(reduction='sum')
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=args.gamma)
            self.tr_epochs = args.tr_epochs
            self.sigmas4eval = args.sigmas4eval
            self.start_epoch = 0
            self.log_interval = args.log_interval
            
            self.alpha = args.alpha
            
        else:
            self.eval_mode()
            self.set_requires_grad([self.net], False)
    
    def eval_mode(self):
        self.net.eval()
    def train_mode(self):
        self.net.train()
        
    def load_networks(self, path):
        self.checkpoint = torch.load(path)
        if len(self.GPU_IDs) == 1:
            self.net.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.net.module.load_state_dict(self.checkpoint['state_dict'])
        print('a pretrained model is loaded from ['+ path +'].....')
    
    def resume_training(self, path, net_only=True):
        self.load_networks(path)
        if not net_only:
            self.start_epoch = self.checkpoint['stop_epoch']
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.last_epoch = self.start_epoch
            print('training hyperparameters are loaded ...')
        else:
            print('only network is loaded!')
    
        
    def _fit_one_epoch(self, train_loader, epoch, logger, adversary=None):
        
        tic_toc = timer()
        for batch_idx, (data_clean, _) in enumerate(train_loader):
            data_clean = data_clean.to(self.device) # 400s
            sigma = (torch.rand(data_clean.shape[0]) * (self.highest_noise_level-self.lowest_noise_level) + \
                self.lowest_noise_level) * 1./255
            data_noisy = add_gaus_noise_to_img(data_clean, sigma)
            
            self.eval_mode()
            self.set_requires_grad([self.net], False)
            data_adv = adversary.perturb(data_noisy, data_clean)
            self.set_requires_grad([self.net], True)
            self.train_mode()
            
            self.optimizer.zero_grad()
            batch_size = data_clean.size()[0]
            output = self.net(data_noisy)
            loss_obs = self.criterion(output, data_clean)
            loss_adv = self.criterion(self.net(data_adv)-output, torch.zeros_like(data_clean))

            loss = ( loss_obs * 1./(1+self.alpha) + loss_adv * self.alpha/(1+self.alpha) ) / (2* batch_size)

            loss.backward()
            self.optimizer.step()
            if (batch_idx+1) % self.log_interval == 0:
                logger.info(f'[{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}, Time for Batches: {tic_toc.toc() :03f}')
        if self.scheduler is not None:
            self.scheduler.step()
    

                
    def fit(self, train_loader, test_loader, logger, adversary_tr=None, adversary_eval=None, save_path='.', paired_loader=None):        
        tic_toc = timer()
        epoch_logger = get_epoch_logger()
        for epoch in range(self.start_epoch, self.tr_epochs):
            logger.info('Training Epoch: {}; Learning rate: {:0.8f}  .....'.format(epoch, self.optimizer.param_groups[0]['lr']))
            self.train_mode()
            self._fit_one_epoch(train_loader, epoch, logger=logger, adversary=adversary_tr)
            psnr_avg_orig, psnr_avg_adv, message = self._evaluate(epoch, test_loader, sigmas=self.sigmas4eval, adversary=adversary_eval)
            logger.info(message)
            logger.info(f'***** PNSR: gaus_avg {psnr_avg_orig}dB / adv_avg {psnr_avg_adv}dB. Time for an epoch: {tic_toc.toc() :.2f}s')
            
            if paired_loader is not None:
                psnr = self.evaluate_pair(paired_loader)
                logger.info(f'***** PSNR ***** paired loader: {psnr:.2f}')
                
            epoch_logger.append_results([epoch, psnr_avg_adv])
            best_epoch = epoch_logger.update_best_epoch_to_logger(logger)
            checkpoint = {'state_dict':self.net.state_dict() if len(self.GPU_IDs)<2 else self.net.module.state_dict(), 
                          'stop_epoch':epoch, 'optimizer': self.optimizer.state_dict()}
            torch.save(checkpoint,os.path.join(save_path, f'ckp_{epoch:02d}.pt'))
            if best_epoch == epoch:
                torch.save(checkpoint, os.path.join(save_path, "ckp_best.pt"))
                
                
    def _evaluate(self, epoch, loader, sigmas=[15, 25], adversary=None, tag='test'):
        PSNR_gaus = []
        PSNR_adv = []
        self.eval_mode(); self.set_requires_grad([self.net], False)
        
        for sigma255 in sigmas:
            # eval gaus
            metric_psnr = PSNR(data_range=1.0)
            metric_psnr.reset()
            for data_clean, _ in loader:
                data_clean = data_clean.to('cuda')  
                sigma_obs = sigma255 * torch.ones([1]) * 1./255
                data_noisy = add_gaus_noise_to_img(data_clean, sigma_obs)
                output = torch.clamp(self.net(data_noisy), min=0, max=1)
                metric_psnr.update([output, data_clean])
            PSNR_gaus.append(metric_psnr.compute())
            # eval adv
            avg_size_dataset = get_avg_size(loader)
            metric_psnr_adv = PSNR(data_range=1.0)
            metric_psnr_adv.reset() 
            for data_clean, _  in loader:
                data_clean = data_clean.to(self.device) # 400s
                ed_sqrt_adv = adversary.eps * 1./math.sqrt(avg_size_dataset)
                sigma_obs = sigma255 * torch.ones([1]) * 1./255 - ed_sqrt_adv
                data_noisy = add_gaus_noise_to_img(data_clean, sigma_obs)
                data_noisy = adversary.perturb(data_noisy, data_clean)
                output_adv = torch.clamp(self.net(data_noisy), min=0, max=1)
                metric_psnr_adv.update([output_adv, data_clean])
            PSNR_adv.append(metric_psnr_adv.compute())
            
        psnr_avg_gaus = np.array(PSNR_gaus).mean()    
        psnr_avg_adv = np.array(PSNR_adv).mean()
        message = '***** PSNR: '
        for i in range(len(sigmas)):
            message += f'Sigma{sigmas[i]}: [gaus {PSNR_gaus[i]:.2f}dB, adv {PSNR_adv[i]:.2f}dB]; '
        return psnr_avg_gaus, psnr_avg_adv, message

    def evaluate_pair(self, loader, num_batch=None):
        self.eval_mode()
        self.set_requires_grad([self.net], False) 
        metric_psnr = PSNR(data_range=1.0)
        metric_psnr.reset()
        idx_batch = 0
        for input, input_path, target, target_path in loader:
            assert target.shape[0] == 1, 'Batch-size should be 1.'
            target = target.to(self.device)
            input = input.to(self.device)
            output = torch.clamp(self.net(input), min=0, max=1)
            metric_psnr.update([output, target])
            idx_batch += 1
            if idx_batch == num_batch:
                break
        psnr_avg = metric_psnr.compute()
        return psnr_avg


def get_avg_size(loader):
    num = 0; size = 0
    for img, _ in loader:
        assert img.shape[0] == 1, 'Batch-size should be 1.'
        num += 1
        size += img.view(-1).shape[0]
    return size / num


if __name__ == '__main__':
    EXP_PATH = os.path.join(args.root, '_'.join([args.exp_name, args.adversary, 'alpha'+str(args.alpha), 'seed'+str(args.SEED)]))
    pathlib.Path(os.path.join(EXP_PATH, 'nets')).mkdir(parents=True, exist_ok=True)
    logger = get_logger(os.path.join(EXP_PATH, 'logging.txt'))
    logger.info(args)
    # Network
    model = Denoiser(args, is_train=True)
    if args.resume:
            model.resume_training(path=args.checkpoint, net_only=args.net_only)
    logger.info(model.net)
    
    # Training Set
    if args.trainset=='Train400':
        trainset = SingleFolder(dir_clean='./data/Train400/clean', ext=',png', is_bin=True, 
                            patch_size=args.patch_size, isAug=True, isScaling=False, repeat=960)
    elif args.trainset == 'Fluorescence':
        trainset = FluorescenceGT(root='./data/denoising-fluorescence/denoising/dataset', train=True, test_fov=[19],
                 repeat=280, patch_size=args.patch_size, isAug=True, isMemory=False)
    elif args.trainset == 'BSD500':
        trainset = SingleFolder(dir_clean='./data/RGB/BSD500/CBSD432', ext='.jpg', is_bin=True, 
                            patch_size=args.patch_size, isAug=True, isScaling=False, repeat=900)
    elif args.trainset == 'SIDDsmall':
        trainset = SIDDpatches(root='./data/RGB/SIDD/Patch512S',
                ndim=3, patch_size=args.patch_size, isAug=True, repeat=40, gt_only=True, is_bin=True)
    else:
        assert False
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle= True, num_workers=8)
    
    # Training adversary
    from advertorch.attacks4IP.zero_mean_pgd import L2PGDAttack
    
        
    if args.adversary == 'eps5_PGD1':
        eps = 5
    elif args.adversary == 'eps1_PGD1':  
        eps = 1
    elif args.adversary == 'eps3_PGD1':  
        eps = 3
    elif args.adversary == 'eps7_PGD1':  
        eps = 7
    else:
        assert False, 'Please use valid attacks...'
        
    if args.network == 'DnCNN-C':
        l2_adv_tr = eps*1./255 * math.sqrt(3 * args.patch_size ** 2 )
    else:
        l2_adv_tr = eps*1./255 * math.sqrt(args.patch_size ** 2)
        
    attack = (L2PGDAttack, dict(loss_fn=nn.MSELoss(), 
                        eps=l2_adv_tr, nb_iter=1, eps_iter=1*l2_adv_tr, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False))
    adversary_tr = attack[0](model.net, **attack[1])
    
    # Eval Set
    if args.testset == 'Set68':
        testset = SingleFolder(dir_clean='./data/Set68/clean', ext='.png', is_bin=True, 
                           patch_size=None, isAug=False, isScaling=False, repeat=1)
    elif args.testset == 'Fluorescence':
        testset = FluorescenceGT(root='./data/denoising-fluorescence/denoising/dataset', train=False, test_fov=[19],
                 repeat=1, patch_size=None, isAug=None, isMemory=False)
    elif args.testset == 'BSD500':
        testset = SingleFolder(dir_clean='./data/RGB/BSD500/CBSD68', ext='.png', is_bin=True, 
                            patch_size=None, isAug=False, isScaling=False, repeat=1)
    elif args.testset == 'SIDDval':
        testset = SIDDsRGBVal(ndim=3, patch_size=None, isAug=False, gt_only=True)
    else:
        assert False
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    
    if args.eval_paired == 'Fluorescence':
        paired_loader = torch.utils.data.DataLoader(FluorescenceTestMix(root='./data/denoising-fluorescence/denoising/dataset', noise_levels=[1]), 
                                                    batch_size=1, shuffle=False, num_workers=4)
    elif args.eval_paired == 'SIDDval':
        paired_loader = torch.utils.data.DataLoader(SIDDsRGBVal(ndim=3, patch_size=None, isAug=False, gt_only=False),
                                                    batch_size=1, shuffle=False, num_workers=4)
    elif args.eval_paired == 'PolyU':
        paired_loader = torch.utils.data.DataLoader(PolyU(), batch_size=1, shuffle=False, num_workers=4)
        
    elif args.eval_paired == 'CC':
        paired_loader = torch.utils.data.DataLoader(CCreal(), batch_size=1, shuffle=False, num_workers=4)
    else:
        paired_loader = None
        
    # Eval adversary
    avg_size_of_testset = get_avg_size(test_loader)
    l2_adv_eval = 5*1./255 * math.sqrt(avg_size_of_testset)
    adversary_eval = L2PGDAttack(model.net, **dict(loss_fn=nn.MSELoss(), 
                        eps=l2_adv_eval, nb_iter=5, eps_iter=0.3*l2_adv_eval, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False,))
    
        
    # Training
    model.fit(train_loader=train_loader, test_loader=test_loader,
                logger=logger, 
                adversary_tr=adversary_tr, adversary_eval=adversary_eval,
                save_path=os.path.join(EXP_PATH, 'nets'), paired_loader=paired_loader)
