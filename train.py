import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from net.losses_extra import SobelEdgeLoss, dark_weighted_l1, color_ratio_loss
import warnings
warnings.filterwarnings("ignore")

opt = option().parse_args()
metrics_file_path = None

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

def append_training_stats(epoch, stats):
    if metrics_file_path is None:
        return
    with open(metrics_file_path, "a") as f:
        f.write(
            "train epoch {}: total={:.4f}, rgb={:.4f}, hvi={:.4f}, edge={:.4f}, dark={:.4f}, color={:.4f}, "
            "hvi_w={:.4f}, edge_w={:.4f}, dark_w={:.4f}, color_w={:.4f}, "
            "k_mean={:.4f}, k_std={:.4f}, k_min={:.4f}, k_max={:.4f}, "
            "k_near_min={:.2%}, k_near_max={:.2%}, lr={:.8f}\n".format(
                epoch,
                stats["total"],
                stats["rgb"],
                stats["hvi"],
                stats["edge"],
                stats["dark"],
                stats["color"],
                stats["hvi_w"],
                stats["edge_w"],
                stats["dark_w"],
                stats["color_w"],
                stats["k_mean"],
                stats["k_std"],
                stats["k_min"],
                stats["k_max"],
                stats["k_near_min"],
                stats["k_near_max"],
                stats["lr"],
            )
        )
    
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    loss_rgb_sum = 0
    loss_hvi_sum = 0
    loss_hvi_weighted_sum = 0
    loss_edge_sum = 0
    loss_dark_sum = 0
    loss_color_sum = 0
    loss_edge_weighted_sum = 0
    loss_dark_weighted_sum = 0
    loss_color_weighted_sum = 0
    k_map_mean_sum = 0
    k_map_std_sum = 0
    k_map_min_sum = 0
    k_map_max_sum = 0
    k_map_low_ratio_sum = 0
    k_map_high_ratio_sum = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            low_rgb = im1 ** gamma
        else:
            low_rgb = im1

        gt_rgb = im2

        # 关键修改：
        # forward 内部已经得到 output_hvi 和输入侧 hvi_aux
        output_rgb, output_hvi, hvi_aux = model(low_rgb, return_hvi=True)
        # 关键修改：
        # GT HVI 使用同一个输入侧 k_map，并且不对 GT 分支反传
        with torch.no_grad():
            gt_hvi = model.HVIT(gt_rgb, aux=hvi_aux)

        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        
        loss_edge = Edge_loss(output_rgb, gt_rgb)
        loss_color = color_ratio_loss(output_rgb, gt_rgb)
        loss_dark = dark_weighted_l1(output_rgb,gt_rgb,low_rgb,opt.dark_alpha)

        #loss = loss_rgb + opt.HVI_weight * loss_hvi
        loss = loss_rgb + opt.HVI_weight * loss_hvi + opt.edge_weight * loss_edge + opt.dark_weight * loss_dark + opt.color_weight * loss_color
        
        iter += 1
        

        optimizer.zero_grad()
        loss.backward()
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        
        optimizer.step()
        
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        loss_rgb_sum += loss_rgb.item()
        loss_hvi_sum += loss_hvi.item()
        loss_hvi_weighted_sum += (opt.HVI_weight * loss_hvi).item()
        loss_edge_sum += loss_edge.item()
        loss_dark_sum += loss_dark.item()
        loss_color_sum += loss_color.item()
        loss_edge_weighted_sum += (opt.edge_weight * loss_edge).item()
        loss_dark_weighted_sum += (opt.dark_weight * loss_dark).item()
        loss_color_weighted_sum += (opt.color_weight * loss_color).item()

        with torch.no_grad():
            k_map = hvi_aux["k_map"]
            k_map_mean_sum += k_map.mean().item()
            k_map_std_sum += k_map.std().item()
            k_map_min_sum += k_map.min().item()
            k_map_max_sum += k_map.max().item()
            k_span = max(model.trans.k_max - model.trans.k_min, 1e-8)
            low_threshold = model.trans.k_min + 0.1 * k_span
            high_threshold = model.trans.k_max - 0.1 * k_span
            k_map_low_ratio_sum += (k_map <= low_threshold).float().mean().item()
            k_map_high_ratio_sum += (k_map >= high_threshold).float().mean().item()

        pic_cnt += 1
        pic_last_10 += 1
        if iter == train_len:
            avg_total = loss_last_10 / pic_last_10
            avg_loss_rgb = loss_rgb_sum / pic_cnt
            avg_loss_hvi = loss_hvi_sum / pic_cnt
            avg_loss_hvi_weighted = loss_hvi_weighted_sum / pic_cnt
            avg_loss_edge = loss_edge_sum / pic_cnt
            avg_loss_dark = loss_dark_sum / pic_cnt
            avg_loss_color = loss_color_sum / pic_cnt
            avg_loss_edge_weighted = loss_edge_weighted_sum / pic_cnt
            avg_loss_dark_weighted = loss_dark_weighted_sum / pic_cnt
            avg_loss_color_weighted = loss_color_weighted_sum / pic_cnt
            avg_k_mean = k_map_mean_sum / pic_cnt
            avg_k_std = k_map_std_sum / pic_cnt
            avg_k_min = k_map_min_sum / pic_cnt
            avg_k_max = k_map_max_sum / pic_cnt
            avg_k_low_ratio = k_map_low_ratio_sum / pic_cnt
            avg_k_high_ratio = k_map_high_ratio_sum / pic_cnt
            print(
                "===> Epoch[{}]: Loss: {:.4f} || "
                "Raw -> RGB: {:.4f} HVI: {:.4f} Edge: {:.4f} Dark: {:.4f} Color: {:.4f} || "
                "Weighted -> HVI: {:.4f} Edge: {:.4f} Dark: {:.4f} Color: {:.4f} || "
                "k(x) mean/std: {:.4f}/{:.4f} min/max: {:.4f}/{:.4f} "
                "near_min: {:.2%} near_max: {:.2%} || "
                "Learning rate: lr={}.".format(
                    epoch,
                    avg_total,
                    avg_loss_rgb,
                    avg_loss_hvi,
                    avg_loss_edge,
                    avg_loss_dark,
                    avg_loss_color,
                    avg_loss_hvi_weighted,
                    avg_loss_edge_weighted,
                    avg_loss_dark_weighted,
                    avg_loss_color_weighted,
                    avg_k_mean,
                    avg_k_std,
                    avg_k_min,
                    avg_k_max,
                    avg_k_low_ratio,
                    avg_k_high_ratio,
                    optimizer.param_groups[0]['lr']
                )
            )
            append_training_stats(
                epoch,
                {
                    "total": avg_total,
                    "rgb": avg_loss_rgb,
                    "hvi": avg_loss_hvi,
                    "edge": avg_loss_edge,
                    "dark": avg_loss_dark,
                    "color": avg_loss_color,
                    "hvi_w": avg_loss_hvi_weighted,
                    "edge_w": avg_loss_edge_weighted,
                    "dark_w": avg_loss_dark_weighted,
                    "color_w": avg_loss_color_weighted,
                    "k_mean": avg_k_mean,
                    "k_std": avg_k_std,
                    "k_min": avg_k_min,
                    "k_max": avg_k_max,
                    "k_near_min": avg_k_low_ratio,
                    "k_near_max": avg_k_high_ratio,
                    "lr": optimizer.param_groups[0]['lr'],
                }
            )
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, pic_cnt
                

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print(f'===> Loading datasets: {opt.dataset}')
    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_v1)
        
    elif opt.dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_blur)

    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_real)
        
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
    
    elif opt.dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_SID)
        
    elif opt.dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        
    elif opt.dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
        test_set = get_fivek_eval_set(opt.data_val_fivek)
    else:
        raise Exception("should choose a dataset")
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    Edge_loss = SobelEdgeLoss().cuda()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 
    training_metrics_dir = os.path.join(opt.val_folder, 'training')
    if not os.path.exists(training_metrics_dir):
        os.mkdir(training_metrics_dir)
        
    global metrics_file_path
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    metrics_file_path = os.path.join(training_metrics_dir, f"metrics{now}.md")
    with open(metrics_file_path, "w") as f:
        f.write("dataset: "+ opt.dataset + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"edge_weight: {opt.edge_weight}\n")
        f.write(f"dark_weight: {opt.dark_weight}\n")
        f.write(f"color_weight: {opt.color_weight}\n")
        f.write(f"dark_alpha: {opt.dark_alpha}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")  
        f.write("\n[train_stats]\n")
        f.write("format: train epoch N: total, raw losses, weighted losses, k(x) stats, lr\n\n")
        f.write("[val_metrics]\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True

            # LOL three subsets
            if opt.dataset == 'lol_v1':
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.dataset == 'lolv2_real':
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.dataset == 'lolv2_syn':
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.dataset == 'lol_blur':
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.dataset == 'SID':
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.dataset == 'SICE_mix':
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.dataset == 'SICE_grad':
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                
            if opt.dataset == 'fivek':
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False

            im_dir = opt.val_folder + output_folder + '*.png'
            is_lol_v1 = (opt.dataset == 'lol_v1')
            is_lolv2_real = (opt.dataset == 'lolv2_real')
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=is_lol_v1, v2=is_lolv2_real, alpha=0.8)
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
            with open(metrics_file_path, "a") as f:
                f.write(f"| {epoch} | { avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")  
        torch.cuda.empty_cache()
