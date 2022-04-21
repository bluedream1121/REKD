import math, torch
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective

from training.model.kernels import Kernels_custom

class KeypointDetectionLoss:
    def __init__(self, args, device):
        custom_kernels = Kernels_custom(args, args.MSIP_sizes)
        kernels = custom_kernels.get_kernels(device) ## with GPU

        self.kernels = kernels

        self.MSIP_sizes = args.MSIP_sizes
        self.MSIP_factor_loss = args.MSIP_factor_loss
        self.patch_size = args.patch_size

    def __call__(self, features_k1, features_k2, h_src_2_dst, h_dst_2_src, mask_borders):
        keynet_loss = 0
        for MSIP_idx, (MSIP_size, MSIP_factor) in enumerate(zip(self.MSIP_sizes, self.MSIP_factor_loss)):
            MSIP_loss = self.ip_loss(features_k1, features_k2, MSIP_size, h_src_2_dst, h_dst_2_src, mask_borders)

            keynet_loss += MSIP_factor * MSIP_loss

            # MSIP_level_name = "MSIP_ws_{}".format(MSIP_size)
            # print("MSIP_level_name {} of MSIP_idx {} : {}, {} ".format(MSIP_level_name, MSIP_idx,  MSIP_loss, MSIP_factor * MSIP_loss)) ## logging

        return keynet_loss

    def ip_loss(self, src_score_maps, dst_score_maps, window_size, h_src_2_dst, h_dst_2_src, mask_borders):

        src_maps, dst_maps, mask_borders = check_divisible(src_score_maps, dst_score_maps, mask_borders, self.patch_size, window_size)

        warped_output_shape =src_maps.shape[2:]

        ## Note that warp_perspective function is not inverse warping! as different with tensorflow.image.transform
        src_maps_warped = warp_perspective(src_maps * mask_borders, h_src_2_dst, dsize=warped_output_shape)
        dst_maps_warped = warp_perspective(dst_maps * mask_borders, h_dst_2_src,  dsize=warped_output_shape)
        visible_src_mask = warp_perspective(mask_borders, h_dst_2_src,  dsize=warped_output_shape) * mask_borders
        visible_dst_mask = warp_perspective(mask_borders, h_src_2_dst,  dsize=warped_output_shape) * mask_borders

        # Remove borders and stop gradients to only backpropagate on the unwarped maps
        src_maps = visible_src_mask * src_maps
        dst_maps = visible_dst_mask * dst_maps
        src_maps_warped = visible_dst_mask * src_maps_warped.detach()
        dst_maps_warped = visible_src_mask * dst_maps_warped.detach()

        # Use IP Layer to extract soft coordinates from original maps & Compute soft weights 
        src_indexes, weights_src = self.ip_layer(src_maps, window_size)
        dst_indexes, weights_dst = self.ip_layer(dst_maps, window_size)   

        # Use argmax layer to extract NMS coordinates from warped maps
        src_indexes_nms_warped = self.grid_indexes_nms_conv(src_maps_warped, window_size)
        dst_indexes_nms_warped = self.grid_indexes_nms_conv(dst_maps_warped, window_size)

        # Multiply weights with the visible coordinates to discard uncommon regions
        weights_src = min_max_norm(weights_src) * max_pool2d(visible_src_mask, window_size)[0]
        weights_dst = min_max_norm(weights_dst) * max_pool2d(visible_dst_mask, window_size)[0]

        loss_src = self.compute_loss(src_indexes, dst_indexes_nms_warped, weights_src, window_size)
        loss_dst = self.compute_loss(dst_indexes, src_indexes_nms_warped, weights_dst, window_size)

        loss_indexes = (loss_src + loss_dst) / 2.

        return loss_indexes
    
    # Obtain soft selected index  (Index Proposal Layer)
    def ip_layer(self, scores, window_size):

        weights, _ = max_pool2d(scores, window_size)

        max_pool_unpool = F.conv_transpose2d(weights, self.kernels['upsample_filter_np_'+str(window_size)], stride=[window_size, window_size])

        exp_map_1 = torch.add(torch.pow(math.e, torch.div(scores, max_pool_unpool+1e-6)), -1*(1.-1e-6))

        sum_exp_map_1 = F.conv2d(exp_map_1, self.kernels['ones_kernel_'+str(window_size)], stride=[window_size, window_size], padding=0) 

        indexes_map = F.conv2d(exp_map_1, self.kernels['indexes_kernel_' + str(window_size)], stride=[window_size, window_size], padding=0)

        indexes_map = torch.div(indexes_map, torch.add(sum_exp_map_1, 1e-6))

        ## compute soft-score
        sum_scores_map_1 = F.conv2d(exp_map_1*scores, self.kernels['ones_kernel_'+str(window_size)], stride=[window_size, window_size], padding=0)

        soft_scores = torch.div(sum_scores_map_1, torch.add(sum_exp_map_1, 1e-6))

        return indexes_map, soft_scores.detach()

    # Obtain hard selcted index by argmax in window
    def grid_indexes_nms_conv(self, scores, window_size):

        weights, indexes = max_pool2d(scores, window_size)

        weights_norm = torch.div(weights, torch.add(weights, torch.finfo(float).eps))

        score_map = F.max_unpool2d(weights_norm, indexes, kernel_size=[window_size, window_size])

        indexes_label = F.conv2d(score_map, self.kernels['indexes_kernel_'+str(window_size)], stride=[window_size, window_size], padding=0)

        # ### To prevent too many the upper-left coordinates cases
        # ind_rand = torch.randint(low=0, high=window_size, size=indexes_label.shape, dtype=torch.int32).to(torch.float32).to(indexes_label.device)

        # indexes_label = torch.where((indexes_label == torch.zeros_like(indexes_label)), ind_rand, indexes_label)
        
        return indexes_label

    @staticmethod
    def compute_loss(src_indexes, label_indexes, weights_indexes, window_size):
        ## loss_ln_indexes_norm 
        norm_sq = torch.pow((src_indexes - label_indexes) / window_size, 2)
        norm_sq = torch.sum(norm_sq, dim=1, keepdims=True)
        weigthed_norm_sq = 1000*(torch.multiply(weights_indexes, norm_sq))
        loss = torch.mean(weigthed_norm_sq)

        return loss


def max_pool2d(scores, window_size):
    ## stride is same as kernel_size as default.
    weights, indexes = F.max_pool2d(scores, kernel_size=(window_size, window_size), padding=0, return_indices=True) 

    return weights, indexes

def check_divisible(src_maps, dst_maps, mask_borders, patch_size, window_size):
    # Check if patch size is divisible by the window size
    if patch_size % window_size > 0:
        batch_shape = src_maps.shape
        new_size = patch_size - (patch_size % window_size)
        src_maps = src_maps[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]
        dst_maps = dst_maps[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]
        mask_borders = mask_borders[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]

    return src_maps, dst_maps, mask_borders

def min_max_norm(A):
    shape = A.shape
    A = torch.flatten(A, start_dim=1)
    A -= A.min(1, keepdim=True)[0]
    A /= (A.max(1, keepdim=True)[0] + 1e-6)
    A = torch.reshape(A, shape)

    return A

