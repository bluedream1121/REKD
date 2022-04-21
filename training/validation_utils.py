import torch, tqdm
import numpy as np

import utils.geometry_tools as geo_tools
from evaluation.evaluation_tools import compute_repeatability_fast
from utils.orientation_tools import compute_orientation_acc
from utils.logger import AverageMeter

def validation_epochs(epoch, dataloader, model, nms_size, device,  num_points=25):

    average_meter = AverageMeter(training=False)
    iterate = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="REKD Validation")
    for idx, batch in iterate:
        images_src, images_dst, h_src_2_dst, h_dst_2_src, sca_src_2_dst, ang_src_2_dst = batch

        features_k1, features_o1 = model(images_src.to(device).type(torch.float32))
        features_k2, features_o2 = model(images_dst.to(device).type(torch.float32))

        ## Compute keypoints store map.
        h_dst_2_src = geo_tools.prepare_homography(h_dst_2_src).to(device)

        mask_src, mask_dst = geo_tools.create_common_region_masks(h_dst_2_src[0].cpu().numpy(), \
            images_src[0].permute(1,2,0).cpu().numpy().shape, images_dst[0].permute(1,2,0).cpu().numpy().shape)

        # Apply NMS
        src_scores = geo_tools.apply_nms(features_k1[0, 0, :, :].cpu().numpy(), nms_size)
        dst_scores = geo_tools.apply_nms(features_k2[0, 0, :, :].cpu().numpy(), nms_size)

        src_scores = np.multiply(src_scores, mask_src)
        dst_scores = np.multiply(dst_scores, mask_dst)

        src_pts = geo_tools.get_point_coordinates(src_scores, num_points=num_points)
        dst_pts = geo_tools.get_point_coordinates(dst_scores, num_points=num_points)

        ## We use the repeateability score following to Superpoint paper appdendix A.
        is_repeatable = compute_repeatability_fast(dst_pts[:, :2], src_pts[:, :2], h_dst_2_src[0].cpu().numpy()) 
        rep_score = sum(is_repeatable) / len(is_repeatable) * 100

        acc, approx_acc = compute_orientation_acc(features_o1, features_o2, ang_src_2_dst.to(device), h_dst_2_src)

        average_meter.update({'repeatability':torch.tensor(rep_score), 'ori_acc':torch.tensor(acc).mean(), 'ori_apx_acc':torch.tensor(approx_acc).mean()})

        msg = average_meter.write_process(idx, len(dataloader), epoch)

        iterate.set_description(msg)

        if idx == 100: 
            break

    # average_meter.write_result(epoch)
    result = average_meter.get_test_results()

    return result['repeatability'], result['ori_acc'], result['ori_apx_acc']


