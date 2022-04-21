import torch
from utils.orientation_tools import align_ori_maps_spatially


def compute_orientation_loss(src_ori, dst_ori, ang_src_2_dst, h_dst_2_src):
    step = 360 // src_ori.shape[1]

    ## spatial alignment
    src_ori, dst_ori_spatial_aligned, mask = align_ori_maps_spatially(src_ori, dst_ori, h_dst_2_src)

    ## histogram alignment
    src_ori_histo_aligned = torch.zeros_like(src_ori)
    for i in range(0, len(ang_src_2_dst)):
        d = int(ang_src_2_dst[i] / step)
        src_ori_histo_aligned[i] = torch.roll(src_ori[i], d, dims=0)
    src_ori_histo_aligned *= mask

    ## calculate loss 
    loss_map = calculate_loss(src_ori_histo_aligned, dst_ori_spatial_aligned)
    loss = loss_map.sum() / mask.sum()

    return loss, loss_map


def calculate_loss(src_hist, dst_hist, eps=1e-7):
    loss_map = -(src_hist * torch.log(dst_hist + eps)).sum(1) 
    
    return loss_map
