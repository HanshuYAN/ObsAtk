from _preamble import *
parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--SEED', default=0, type=int)
parser.add_argument('--GPU_IDs', nargs='+', default=[0], type=int)


parser.add_argument('--attack', type=str, choices=['obs', 'ins', 'random'],
                    default='obs', 
                    # default='random', 
                    )

# parser.add_argument('--testset', default='Set12')
# # parser.add_argument('--testset', default='Set68')
# parser.add_argument('--network', type=str, default='DnCNN-B')
# parser.add_argument('--checkpoints', type=list,
#                     default=[
#                         # ['./experiments/Gxx25_Tr400_DnCNN_NT/nets/ckp_best.pt', None,],
#                         # ['./experiments/Gxx25_Tr400_DnCNN_NT2/nets/ckp_best.pt', None,],
#                         # ['./experiments/Gxx25_Tr400_DnCNN_NT3/nets/ckp_best.pt', None,],
                        
                        
#                         # ['./experiments/Gxx25_Tr400_DnCNN_vAT2_eps5_PGD1/nets/ckp_best.pt', None,],
#                         # ['./experiments/Gxx25_Tr400_DnCNN_vAT3_eps5_PGD1/nets/ckp_best.pt', None,],
#                         # ['./experiments/Gxx25_Tr400_DnCNN_vAT4_eps5_PGD1/nets/ckp_best.pt', None,],
                        
#                         # ['./experiments/Gxx25_Tr400_DnCNN_htAT_eps5_PGD1_alpha0.2/nets/ckp_best.pt', None,],
#                         # ['./experiments/Gxx25_Tr400_DnCNN_htAT2_eps5_PGD1_alpha0.2/nets/ckp_best.pt', None,],
#                         ['./experiments/Gxx25_Tr400_DnCNN_htAT3_eps5_PGD1_alpha0.5/nets/ckp_best.pt', None,],
#                         ])


parser.add_argument('--testset', default='BSD500')
# parser.add_argument('--testset', default='Kodak24')
parser.add_argument('--network', type=str, default='DnCNN-C')
parser.add_argument('--checkpoints', type=list,
                    default=[
                        # ['./experiments/Gxx25_BSD500_DnCNN_NT/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_NT2/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_NT3/nets/ckp_best.pt', None,],
                        
                        # ['./experiments/Gxx25_BSD500_DnCNN_vAT_eps5_PGD1/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_vAT2_eps5_PGD1/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_vAT4_eps5_PGD1/nets/ckp_best.pt', None,],
                        
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT_eps5_PGD1_alpha1.0/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT2_eps5_PGD1_alpha1.0/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT3_eps5_PGD1_alpha1.0/nets/ckp_best.pt', None,],
                        
                        
                        # # abalation
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT_eps5_PGD1_alpha0.1/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT_eps5_PGD1_alpha0.25/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT_eps5_PGD1_alpha1.0/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT_eps5_PGD1_alpha2.0/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT_eps5_PGD1_alpha5.0/nets/ckp_best.pt', None,],
                        # ['./experiments/Gxx25_BSD500_DnCNN_htAT_eps5_PGD1_alpha7.0/nets/ckp_best.pt', None,],
                        ['./experiments/Gxx25_BSD500_DnCNN_htAT_eps5_PGD1_alpha10.0/nets/ckp_best.pt', None,],
                        ])


args, _ = parser.parse_known_args()
np.random.seed(args.SEED)
torch.manual_seed(args.SEED)

import datasets.misc as misc
from utils import makedirs, get_logger
from datasets.image_processing import *
from datasets.misc import adding_guas_noise, adding_poisson_noise, adding_uniform_noise
from models.BaseModel import BaseModelDNN
from models.nets.DnCNN import DnCNN

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
    def __init__(self, args=None, device='cuda') -> None:
        super().__init__()
        self.net = Network().to(device)
        self.GPU_IDs = args.GPU_IDs
        if len(self.GPU_IDs) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=self.GPU_IDs)        
        self.eval_mode()
        self.set_requires_grad([self.net], False)
            
def get_avg_size(loader):
    num = 0; size = 0
    for img, _ in loader:
        assert img.shape[0] == 1, 'Batch-size should be 1.'
        num += 1
        size += img.view(-1).shape[0]
    return size / num

def Evaluate(predict, loader, advperturb1=None, advperturb2=None, tag='gaus', num_batch=None, save_path=None, device=torch.device("cuda:0")):
    if save_path is not None:
        makedirs(os.path.join(save_path, tag+'_out'))
        makedirs(os.path.join(save_path, tag+'_in'))
        makedirs(os.path.join(save_path, tag+'_noise'))
    
    metric_psnr = PSNR(data_range=1.0)
    metric_psnr.reset()
    metric_dist = Average()
    metric_dist.reset()
    
    metric_energy_density = Average()
    metric_energy_density.reset()
    
    idx_batch = 0
    for target, target_path in tqdm.tqdm(loader):
        assert target.shape[0] == 1, 'Batch-size should be 1.'
        target = target.to(device)
        input = advperturb1(target, target)
        if advperturb2 is not None:
            input = advperturb2(input, target)
        
        output = torch.clamp(predict(input), min=0, max=1)
        metric_psnr.update([output, target])

        adv_noise = target-input
        distance = torch.norm((adv_noise).view(1,-1), p=2)
        metric_dist.update(distance)
        
        energy_density = distance ** 2 / target.view(-1).shape[0]
        metric_energy_density.update(energy_density)
        
        if save_path is not None:
            filename = target_path[0].split('/')[-1].split('.')[0] + '.png'
            skimage.io.imsave(os.path.join(save_path, tag+'_out', filename), skimage.img_as_ubyte(misc.Tensor2Img(output[0])))
            skimage.io.imsave(os.path.join(save_path, tag+'_in', filename), skimage.img_as_ubyte(misc.Tensor2Img(input[0])))
            # print(adv_noise.mean())
            noise_map = adv_noise[0]-adv_noise[0].mean()+0.5
            skimage.io.imsave(os.path.join(save_path, tag+'_noise', filename), skimage.img_as_ubyte(misc.Tensor2Img(torch.clamp(noise_map, 0, 1))))
        idx_batch += 1
        if idx_batch == num_batch:
            break
    psnr_avg = metric_psnr.compute()
    dist_avg = metric_dist.compute()
    energy_density_avg = metric_energy_density.compute()
    return psnr_avg, dist_avg, energy_density_avg



            
if __name__ == '__main__':
    logger = get_logger(os.path.join('./results', 'logging.txt'))
    logger.info(args)

    model = Denoiser(args)
    # gray
    if args.testset == 'Set68':
        testset = SingleFolder(dir_clean='./data/Set68/clean', ext='.png', is_bin=False, 
                           patch_size=None, isAug=False, isScaling=False, repeat=1)
    elif args.testset == 'Set12':
        testset = SingleFolder(dir_clean='./data/Set12/clean', ext='.png', is_bin=False, 
                           patch_size=None, isAug=False, isScaling=False, repeat=1)
    # rgb
    elif args.testset == 'BSD500':
        testset = SingleFolder(dir_clean='./data/RGB/BSD500/CBSD68', ext='.png', is_bin=False, 
                            patch_size=None, isAug=False, isScaling=False, repeat=1)
    elif args.testset == 'Kodak24':
        testset = SingleFolder(dir_clean='./data/RGB/Kodak24', ext='.png', is_bin=False, 
                            patch_size=None, isAug=False, isScaling=False, repeat=1)
    else:
        assert False
        
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)    
    avg_size_of_testset = get_avg_size(test_loader)
    
    from advertorch.attacks4IP.zero_mean_pgd import L2PGDAttack
    # sigmas = [25, 15, 10]
    sigmas = [25, 15]
    for ckp, save_path in args.checkpoints:
        logger.info('\n')
        logger.info(ckp)
        model.load_networks(ckp)


        if args.attack == 'obs':
            for sigma255 in sigmas:
                # adv_sigmas = [5,7]
                # adv_sigmas = [3,5,7]
                adv_sigmas = [5]
                for sigma255_adv in adv_sigmas:
                    sigma = sigma255 * 1./255
                    l2_budget_vs_gaus = sigma * math.sqrt(avg_size_of_testset)
                    print(f'obsAttack, \sigma [{sigma255}/255={sigma:.3f}], ED [{sigma**2:.5f}], l2_norm [{l2_budget_vs_gaus :.5f}] with avg size {avg_size_of_testset}')
                    
                    sigma_obs = (sigma255-sigma255_adv)*1./255
                    l2_obs = sigma_obs * math.sqrt(avg_size_of_testset)
                    l2_adv = sigma255_adv*1./255 * math.sqrt(avg_size_of_testset)
                    print(f'obs noise: \sigma [{sigma255-sigma255_adv}/255={sigma_obs:.3f}], l2_norm[{l2_obs}] ; adv: \sigma [{sigma255_adv}/255={sigma255_adv/255.:.3f}], l2_budget [{l2_adv}] ')
                    
                    noiser = adding_guas_noise(sigma_obs)
                    lst_attack = [
                        (L2PGDAttack, dict(loss_fn=nn.MSELoss(), 
                            eps=l2_adv, nb_iter=5, eps_iter=0.3*l2_adv, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)),
                    ]
                    for attack_class, attack_kwargs in lst_attack:
                        adversary = attack_class(model.net, **attack_kwargs)
                        psnr, dist, ed = Evaluate(model.net, test_loader, advperturb1=noiser.perturb, advperturb2=adversary.perturb, tag=f'advObs_ed{sigma255}_{sigma255_adv}',
                                            num_batch=None, 
                                            save_path=save_path,
                                            )
                        logger.info(attack_class.__name__ + f', ***** PNSR {psnr:.2f}, real noise: ED {ed:.5f}, Distance {dist}. \n')
                
                

        if args.attack == 'random':
            for sigma255 in sigmas:
                sigma = sigma255 * 1./255
                print(f'Gaussian [{sigma255}], ED [{sigma**2:.5f}] .....')
                
                # Guassian
                l2_budget = sigma * math.sqrt(avg_size_of_testset)
                print(f'Gaus [{sigma255}/255={sigma:.3f}], l_2 budget [{l2_budget :.4f} with avg size {avg_size_of_testset}]')
                
                adversary = adding_guas_noise(sigma)
                psnr, dist, ed = Evaluate(model.net, test_loader, advperturb1=adversary.perturb, advperturb2=None, tag=f'gaus_ed{sigma255}',
                                    num_batch=None,
                                    save_path=save_path,
                                    )
                logger.info(f'***** PNSR {psnr:.2f}, real noise: ED {ed:.5f} , Distance {dist:.3f}  \n')
                

                # uniform
                u =  sigma * math.sqrt(3)
                l2_budget = (u*1./math.sqrt(3)) * math.sqrt(avg_size_of_testset)
                print(f'Uniform [{sigma255}*\sqrt(3)/255={u:.3f}] with l_2 budget [{l2_budget :.4f} with avg size {avg_size_of_testset}]')
                
                adversary = adding_uniform_noise(u)
                psnr, dist, ed = Evaluate(model.net, test_loader, advperturb1=adversary.perturb, advperturb2=None, tag=f'uniform_ed{sigma255}',
                                    num_batch=None,
                                    save_path=save_path,
                                    )
                logger.info(f'***** PNSR {psnr:.2f}, real noise: ED {ed:.5f} , Distance {dist:.3f} \n')
                
                

                """
                # possion
                
                lam = (sigma*255) ** 2
                l2_budget = math.sqrt(lam) * math.sqrt(avg_size_of_testset) / 255
                print(f'Poisson [{lam:.1f}-255] with l_2 budget [{l2_budget :.4f}]')
                
                adversary = adding_poisson_noise(lam)
                psnr, dist, ed = Evaluate(model.net, test_loader, advperturb1=adversary.perturb, advperturb2=None, tag=f'poisson_ed{sigma255}',
                                    num_batch=None,
                                    # save_path=args.save_path,
                                        save_path=save_path,
                                    )
                logger.info(f'***** PNSR {psnr:.2f}, Poisson [{lam:.1f}-255], ED {ed:.5f} , Distance {dist:.3f} with size {avg_size_of_testset} \n')
                """
                
                
    """[summary]
    
        if args.attack == 'ins':
            for sigma255 in sigmas:
                sigma = sigma255 * 1./255
                l2_adv = sigma * math.sqrt(avg_size_of_testset)
                print(f'Ins-based attack... Gaus [{int(sigma255)}/{255}]: ED [{sigma**2:.5f}], adv budget l_2 [{l2_adv :.5f}]')
                
                lst_attack = [
                    # (L2PGDAttack, dict(loss_fn=nn.MSELoss(), 
                    #     eps=l2_adv, nb_iter=1, eps_iter=1*l2_adv, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)),
                    (L2PGDAttack, dict(loss_fn=nn.MSELoss(), 
                        eps=l2_adv, nb_iter=5, eps_iter=0.3*l2_adv, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)),
                ]
                for attack_class, attack_kwargs in lst_attack:
                    adversary = attack_class(model.net, **attack_kwargs)
                    psnr, dist, ed = Evaluate(model.net, test_loader, advperturb1=adversary.perturb, tag=f'advIns_ed{sigma255}',
                                        num_batch=None, 
                                        save_path=save_path,
                                        )
                    logger.info(attack_class.__name__ + f', ***** PNSR {psnr}, ED {ed:.5f}, Distance {dist}. \n')
    """
