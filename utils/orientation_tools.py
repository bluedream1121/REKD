import torch
from kornia.geometry.transform import warp_perspective

def align_ori_maps_spatially(src_ori, dst_ori, h_dst_2_src):
    ## rotate target image toward source image
    # warning : angle in rotate function is counter clockwise
    _, _, sh, sw = src_ori.shape
    dst_ori_aligned = warp_perspective(dst_ori, h_dst_2_src, (sh, sw), align_corners=True) 

    ## make mask for rotated histogram channel wise. For count the number of valid pixels.
    # valid: 1, invalid: 0
    b, c, dh, dw = dst_ori.shape
    mask = torch.ones(b, 1, dh, dw).to(src_ori.device)
    mask = warp_perspective(mask, h_dst_2_src, (sh, sw), align_corners=True)
    mask = mask > 0  ##  valid: 1, invalid: 0

    return src_ori, dst_ori_aligned, mask

def compute_orientation_acc(src_ori, dst_ori, ang_src_2_dst, h_dst_2_src, tolerance=1):
    ## src_ori/dst_ori : (B, G, H, W), ang_src_2_dst: (B, 1), h_dst_2_src: (B, 3, 3)

    B, G, _, _  = src_ori.shape
    step = 360 // G   
    ori_threshold = step * tolerance  ## for approx acc (error difference one bin tolerance.)
    d_threshold = int(ori_threshold // step) 

    ## Compute the pred angle difference
    src_ori, dst_ori_aligned, mask = align_ori_maps_spatially(src_ori, dst_ori, h_dst_2_src)
    src_ang, dst_ang = _histograms_to_angle_values(src_ori, dst_ori_aligned)

    pred_ang_diff = ((dst_ang - src_ang + G) % G) * mask.squeeze(1)
    pred_ang_diff = torch.flatten(pred_ang_diff, start_dim=1)
    mask = torch.flatten(mask, start_dim=1)  ## invalid region mask

    ## Compute the accuracies.
    total_cnt = mask.sum(1)
    correct_cnt = _compute_correct_orientations(pred_ang_diff, ang_src_2_dst, mask, step, G)

    approx_cnt = 0
    for i in range(-d_threshold, d_threshold+1):
        approx_cnt += _compute_correct_orientations(pred_ang_diff, ang_src_2_dst, mask, step, G, i)

    return correct_cnt/total_cnt, approx_cnt/total_cnt


def _histograms_to_angle_values(src_ori, dst_ori_aligned):
    src_ang = obtain_orientation_value_map(src_ori, mode='argmax') 
    dst_ang = obtain_orientation_value_map(dst_ori_aligned, mode='argmax')  
    return src_ang, dst_ang


def _compute_correct_orientations(pred_ang_diff, GT_ang_diff, mask, step, G, toler=0):
    ## G is orientation bin size. (the order of group)
    GT_bin_diff = ((GT_ang_diff // step) + G + toler) % G
    GT_bin_diff = mask * GT_bin_diff
    return (torch.where(pred_ang_diff ==  GT_bin_diff, 1, 0) * mask ).sum(1)


########### for inference

def obtain_orientation_value_map(feat, mode='argmax'):
    ## input : B, C, H, W,  output : B, H, W
    B, C, H, W = feat.shape
    
    if mode == 'argmax':
        res = torch.argmax(feat, dim=1)
    elif mode == 'soft_argmax':
        index_kernel = torch.linspace(0, C-1, C).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(feat.device)
        # res = torch.mul(torch.softmax(feat,dim=1), index_kernel)
        res = torch.mul(feat, index_kernel)
        res = res.mean(dim=1)

    return res


