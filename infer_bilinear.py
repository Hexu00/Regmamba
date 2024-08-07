import glob
import os, utils
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import time
#import summary
from Mamba_Net_3_19 import *

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=3.0,
                    help="smth_labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=403,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=2,
                    help="using l2 or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
using_l2 = opt.using_l2

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = SpatialTransform().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    atlas_dir = '/home/huxin/Dataset/IXI_data/atlas.pkl'
    test_dir = '/home/huxin/Dataset/IXI_data/Test/'

    model_idx = -3
    model_dir = './IXI_mamba3_19_diff_L2ss_{}_Chan_{}_LR_{}_Smooth_{}/'.format(using_l2, start_channel, lr, smooth)
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+model_dir[:-1]+'_Test.csv'):
        os.remove('Quantitative_Results/'+model_dir[:-1]+'_Test.csv')
    csv_writter(model_dir[:-1], 'Quantitative_Results/' + model_dir[:-1]+'_Test')
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + model_dir[:-1]+'_Test')

    model = Mamba_UNet(2, 3, start_channel,blocks=2).cuda()
    
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location='cuda:0')#['state_dict']
    model.load_state_dict(best_model)
    model.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"模型参数数量: {num_params}")
    # reg_model = utils.register_model(config.img_size, 'nearest')
    # reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),trans.NumpyType((np.float32, np.int16)),])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    Reg_time = []
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            start_time = time.time()
            f_xy = model(x.float().to(device), y.float().to(device))
            reg_time = time.time() - start_time
            Reg_time.append(reg_time)
            D_f_xy = diff_transform(f_xy)  #f_xy #

            X_Y = transform(x, D_f_xy.permute(0, 2, 3, 4, 1))

            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            #x_segs = model.spatial_trans(x_seg.float(), flow.float())
            x_segs = []
            for i in range(46):  # 目的是对输入的分割图像序列进行配准，输出得到 def_out
                def_seg = transform(x_seg_oh[:, i:i + 1, ...].float().to(device), D_f_xy.permute(0, 2, 3, 4, 1))
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            
            dd, hh, ww = D_f_xy.shape[-3:]
            D_f_xy = D_f_xy.detach().cpu().numpy()
            D_f_xy[:,0,:,:,:] = D_f_xy[:,0,:,:,:] * dd / 2
            D_f_xy[:,1,:,:,:] = D_f_xy[:,1,:,:,:] * hh / 2
            D_f_xy[:,2,:,:,:] = D_f_xy[:,2,:,:,:] * ww / 2

            jac_det = utils.jacobian_determinant_vxm(D_f_xy[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line +','+ str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + model_dir[:-1]+'_Test')
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1 
     
#values = [0.757, 0.762, 0.789, 0.732, 0.745]
# Calculate mean and standard deviation
#mean_value = np.mean(values)   
#std_dev = np.std(values, ddof=1) 

        print('Deformed DSC: {:.4f} +- {:.4f}, Affine DSC: {:.4f} +- {:.4f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))

        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

        print('Deformed time: {:.4f} +- {:.4f}'.format(torch.mean(torch.Tensor(Reg_time)),
                                                       torch.std(torch.Tensor(Reg_time))))



def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    main()