## This is merged files extract_multiscale_features.py and evaluation.py
import os, cv2,  torch, tqdm
import numpy as np
import torch.nn.functional as F
from skimage.transform import pyramid_gaussian

from training.model.load_models import load_detector, load_descriptor
from utils.orientation_tools import obtain_orientation_value_map
import utils.geometry_tools as geo_tools
import utils.desc_aux_function as desc_aux
from .evaluation_tools import *


class MultiScaleFeatureExtractor:
    def __init__(self, args, exp_name, split, dataset_dir='datasets'):

        ## configurations
        self.num_points=args.num_points
        self.pyramid_levels=args.pyramid_levels
        self.upsampled_levels=args.upsampled_levels

        self.border_size=args.border_size
        self.nms_size=args.nms_size
        self.desc_scale_factor= 2.0
        self.scale_factor_levels=np.sqrt(2)

        self.image_list, _ = load_hpatches_images(dataset_dir, split)

        self.save_feat_dir = make_save_dir(exp_name)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model1 = load_detector(args, device)
        self.model2 = load_descriptor(args, device)

        ## points level define (Image Pyramid level)
        point_level = []
        tmp = 0.0
        factor_points = (self.scale_factor_levels ** 2)
        self.levels = self.pyramid_levels + self.upsampled_levels + 1
        for idx_level in range(self.levels):
            tmp += factor_points ** (-1 * (idx_level - self.upsampled_levels))
            point_level.append(self.num_points * factor_points ** (-1 * (idx_level - self.upsampled_levels)))

        self.point_level = np.asarray(list(map(lambda x: int(x/tmp)+1, point_level)))

        ## GPU
        self.device = device

        print("Descriptor extraction method : ", args.descriptor)
        print('Extract features at : {}'.format(self.save_feat_dir))


    def extract_hpatches(self, model1=None):
        if self.model1 is not None and model1 is None:  ## if not use pre-defined model.
            model1 = self.model1

        with torch.no_grad():
            iterate = tqdm.tqdm(self.image_list, total=len(self.image_list))

            for path in iterate:

                im_pts, descriptors, ori_values = self._extract_features(path, model1)

                self._save_features(path, im_pts, descriptors, ori_values)

                msg = "Extract {}".format('/'.join(path.split('/')[-2:]) )
                iterate.set_description(msg)


    @staticmethod
    def read_image(path):
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)[np.newaxis, :, :]  ## (1, H, W)
        im = im.astype(float) / im.max()
        return im

    def _extract_features(self, path, model1):
        image = self.read_image(path)

        score_maps, ori_maps = self._compute_score_maps(image, model1)

        im_pts = self._estimate_keypoint_coordinates(score_maps)

        im_pts, ori_values = self._estimate_orientations(im_pts, ori_maps)

        descriptors = self._extract_descriptors(image, im_pts)

        return im_pts, descriptors, ori_values


    def _compute_score_maps(self, image, model1):
        pyramid = pyramid_gaussian(image, max_layer=self.pyramid_levels, downscale=self.scale_factor_levels)
        up_pyramid = upsample_pyramid(image, upsampled_levels=self.upsampled_levels, scale_factor_levels=self.scale_factor_levels)

        score_maps = {}
        ori_maps = {}
        for (j, down_image) in enumerate(pyramid):  ## Pyramid is downsampling images.
            key_idx = j + 1 + self.upsampled_levels
            score_maps, ori_maps =  self._obtain_feature_maps(model1, down_image, key_idx, score_maps, ori_maps)

        if self.upsampled_levels:
            for j, up_image in enumerate(up_pyramid):  ## Upsample levels is for upsampling images.
                key_idx = j + 1 
                score_maps, ori_maps =  self._obtain_feature_maps(model1, up_image, key_idx, score_maps, ori_maps)

        return score_maps, ori_maps


    def _obtain_feature_maps(self, model1, im, key_idx, score_maps, ori_maps):

        im = torch.tensor(im).unsqueeze(0).to(torch.float32).cuda()
        im_scores, ori_map = model1(im)
        im_scores = geo_tools.remove_borders(im_scores[0,0,:,:].cpu().detach().numpy(), borders=self.border_size)

        score_maps['map_' + str(key_idx)] = im_scores
        ori_maps['map_' + str(key_idx)] = ori_map

        return score_maps, ori_maps


    def _estimate_keypoint_coordinates(self, score_maps):
        im_pts = []
        for idx_level in range(self.levels):
            scale_value = (self.scale_factor_levels ** (idx_level - self.upsampled_levels))
            scale_factor = 1. / scale_value

            h_scale = np.asarray([[scale_factor, 0., 0.], [0., scale_factor, 0.], [0., 0., 1.]])
            h_scale_inv = np.linalg.inv(h_scale)
            h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

            num_points_level = self.point_level[idx_level]
            if idx_level > 0:
                res_points = int(np.asarray([self.point_level[a] for a in range(0, idx_level + 1)]).sum() - len(im_pts))
                num_points_level = res_points

            ## to make the output score map derive more keypoints
            score_map = score_maps['map_' + str(idx_level + 1)]

            im_scores = geo_tools.apply_nms(score_map, self.nms_size)
            im_pts_tmp = geo_tools.get_point_coordinates(im_scores, num_points=num_points_level)
            im_pts_tmp = geo_tools.apply_homography_to_points(im_pts_tmp, h_scale_inv)

            if not idx_level:
                im_pts = im_pts_tmp
            else:
                im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)

        im_pts = im_pts[(-1 * im_pts[:, 3]).argsort()]
        im_pts = im_pts[:self.num_points]

        return im_pts

    ## convert orientation histogram to orientation values
    def _estimate_orientations(self, im_pts, ori_maps):
        ori_map = ori_maps['map_' + str(1 + self.upsampled_levels)]  ## original_image_size

        bin_size, h, w = ori_map.shape[1:]
        im_pts = check_coorinates_in_image(im_pts, w, h)

        ori_values = obtain_orientation_value_map(ori_map, mode='argmax')  * 360 / bin_size ## to degree

        im_pts = assign_orientations_to_keypoints(im_pts, ori_values)

  
        return im_pts, ori_values


    def _extract_descriptors(self, image, im_pts):
        descriptors = []

        for idx_desc_batch in range(int(len(im_pts) / 10000 + 1)):
            points_batch = im_pts[idx_desc_batch * 10000: (idx_desc_batch + 1) * 10000]

            if not len(points_batch):
                break

            kpts_coord = torch.tensor(points_batch[:, :2]).to(torch.float32).cuda()
            kpts_batch = torch.zeros(len(points_batch)).to(torch.float32).cuda()
            input_image = torch.tensor(image).unsqueeze(0).to(torch.float32).cuda()
            kpts_scale = torch.tensor(points_batch[:, 2] * self.desc_scale_factor).to(torch.float32).cuda()

            patch_batch = desc_aux.build_patch_extraction(kpts_coord, kpts_batch, input_image, kpts_scale)
            patch_batch = torch.reshape(patch_batch, (patch_batch.shape[0], 1, 32, 32))

            data_a = patch_batch.to(self.device)

            with torch.no_grad():
                out_a = self.model2(data_a)

            desc_batch = out_a.data.reshape(-1, 128)
            if idx_desc_batch == 0:
                descriptors = desc_batch
            else:
                descriptors = torch.cat([descriptors, desc_batch], axis=0)

        return descriptors.cpu().numpy()

    def _save_features(self, path, im_pts, descriptors, ori_values, save_ori_map=False):

        create_result_dir(os.path.join(self.save_feat_dir, path))

        file_name = os.path.join(self.save_feat_dir, path)+'.kpt'
        np.save(file_name, im_pts)

        file_name = os.path.join(self.save_feat_dir, path)+'.dsc'
        np.save(file_name, descriptors)

        if save_ori_map:
            ori_map = ori_values.squeeze().cpu().numpy()  
            file_name = os.path.join(self.save_feat_dir, path) + '.orimap'
            np.save(file_name, ori_map)


    def get_save_feat_dir(self):
        return self.save_feat_dir 