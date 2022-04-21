import tqdm, torch
import numpy as np
## Loss function.
from training.loss.score_loss_function import KeypointDetectionLoss
from training.loss.orientation_loss_function import compute_orientation_loss

import utils.geometry_tools as geo_tools
from utils.logger import AverageMeter



def training_epochs(epoch, dataloader, model, optimizer, args, device):

    msip_loss_function = KeypointDetectionLoss(args, device)

    average_meter = AverageMeter(training=True)

    iterate = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="REKD Training")
    for idx, batch in iterate:
        images_src, images_dst, h_src_2_dst, h_dst_2_src, sca_src_2_dst, ang_src_2_dst = batch

        features_k1, features_o1 = model(images_src.to(device).type(torch.float32))
        features_k2, features_o2 = model(images_dst.to(device).type(torch.float32))

        h_src_2_dst = geo_tools.prepare_homography(h_src_2_dst).to(device)
        h_dst_2_src = geo_tools.prepare_homography(h_dst_2_src).to(device) 

        ## Compute loss 1 : orientation loss
        ori_loss, ori_loss_map = compute_orientation_loss(features_o1, features_o2, ang_src_2_dst.to(device), h_dst_2_src)
        ori_loss = ori_loss * args.ori_loss_balance

        ## Compute loss 2 : keynet loss
        mask_borders = get_border_masks(images_src).to(device)

        keynet_loss  = msip_loss_function(features_k1, features_k2, h_src_2_dst, h_dst_2_src, mask_borders)

        loss = ori_loss + keynet_loss

        isnan(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## logging
        average_meter.update({"key_loss": keynet_loss, "ori_loss": ori_loss, "total_loss": loss  })
        msg = average_meter.write_process(idx, len(dataloader), epoch)
        iterate.set_description(msg)


    # average_meter.write_result(epoch)
    result = average_meter.get_results()

    return result['key_loss'], result['ori_loss'], result['total_loss']

def get_border_masks(images_src):
    b,c,h,w = images_src.shape
    input_border_mask = geo_tools.remove_borders(torch.ones([b,1,h,w]), 16)
    return input_border_mask

def isnan(loss):
    if torch.isnan(loss):
        print("Loss nan!!")
        exit()