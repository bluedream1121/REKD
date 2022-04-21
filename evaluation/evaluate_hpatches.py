import os, torch,  time,  tqdm
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

from .evaluation_tools import *
from utils.matchers import mnn_matcher


class HPatchesEvaluator:
    def __init__(self, hpatches_path, split, nfeatures, outlier_rejection, outlier_threshold=30):

        _, self.seqs= load_hpatches_images('datasets', split)

        self.nfeatures = nfeatures
        self.outlier_rejection = outlier_rejection
        self.outlier_threshold = outlier_threshold
        self.correct_threshold = 5

        self.columns = ["src", "dst", "total_points", "repeatability_score", "pred. match", "correct@1", "MMA@1", "correct@3", "MMA@3", "correct@5", "MMA@5"]
        self.logger = HPatchesAverageMeter()
        self.logger_illum = HPatchesAverageMeter()
        self.logger_view = HPatchesAverageMeter()        


    def compute(self, load_data):

        iterate = tqdm.tqdm(sorted(list(self.seqs)), total=len(self.seqs))

        for seq in iterate:
            seq_name = seq.split('/')[-1]

            iterate.set_description("Sequence: {}".format(seq_name))
            
            for tar_idx in range(2, 7):
                ref_path = seq+'/1.ppm'
                tar_path = seq+'/'+str(tar_idx)+'.ppm'

                k1, o1, s1, d1 = self.load_features(load_data, ref_path, nfeatures=self.nfeatures)
                k2, o2, s2, d2 = self.load_features(load_data, tar_path, nfeatures=self.nfeatures)

                if k1 is None or k2 is None:
                    print("No keypoints")
                    continue
                
                H = GetGroundTruth(ref_path, tar_path)

                ## compute the average number of points
                total_points = (len(d1) + len(d2)) / 2 
                
                ## compute k1 -> k2 repeatability.
                repeatable = compute_repeatability_fast(k1[:, :2], k2[:, :2], H, pixel_threshold=3)
                repeatability_score = sum(repeatable) / len(repeatable) * 100
                
                ## compute matching results
                matches = mnn_matcher(torch.tensor(d1).cuda(), torch.tensor(d2).cuda())

                if self.outlier_rejection:
                    matches = self.reject_outlier_by_orientation(o1, o2, matches, threshold=self.outlier_threshold)
                    
                match_cnt, correct_cnt, precision = self.compute_results(k1[matches[:, 0], :2], k2[matches[:, 1], :2], H, corr_thres=self.correct_threshold)
                

                logging_data = (seq_name, tar_idx, total_points, repeatability_score, match_cnt, correct_cnt, precision)

                self.logger.log_data(logging_data)
                if seq_name[0] == 'i':
                    self.logger_illum.log_data(logging_data)
                if seq_name[0] == 'v':
                    self.logger_view.log_data(logging_data)

        return self.print_pandas_results()



    def load_features(self, feature_path, impath, nfeatures):

        post_fix = "/".join(impath.split('/')[-2:])[:-3]
        kpts_path = os.path.join(feature_path, post_fix + 'ppm.kpt.npy')
        desc_path = os.path.join(feature_path, post_fix + 'ppm.dsc.npy')

        kpts = np.load(kpts_path)
        desc = np.load(desc_path)

        kpts = kpts[:nfeatures]
        desc = desc[:nfeatures]

        s = None if kpts.shape[1] < 3 else kpts[:, 2]
        o = None if kpts.shape[1] < 5 else kpts[:, 4]
        
        return kpts, o, s, desc


    def reject_outlier_by_orientation(self, o1, o2, matches, threshold=30):
        diff = (o2[matches[:, 1]] - o1[matches[:, 0]] + 360) % 360

        counter = Counter(np.round(diff))
        ori = counter.most_common(1)[0][0]

        ori_min = (ori - threshold + 360) % 360
        ori_max = (ori + threshold + 360) % 360

        if (ori_min > ori_max):
            indices = np.where(((diff >= ori_min) & (diff < 360)) | ((diff <= ori_max) & (diff >= 0)))
        else:
            indices = np.where((diff >= ori_min) & (diff <= ori_max))

        return matches[indices]
    
    
    def compute_results(self, k1, k2, H, corr_thres=5):
        warp_points = warpPerspectivePoints(k1, H) ## warp kpts1 to image2 (using GT)
        gt_k2 = warp_points

        match_cnt = k1.shape[0]

        correct_cnt = np.zeros(corr_thres)
        precision = np.zeros(corr_thres)

        for (x1, y1), (x2, y2) in zip(k2, gt_k2):
            distance = torch.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
            for pixel_thres in range(corr_thres):
                if distance <= pixel_thres + 1 :
                    correct_cnt[pixel_thres] += 1

        for pixel_thres in range(corr_thres):
            precision[pixel_thres] = 0 if match_cnt == 0 else correct_cnt[pixel_thres] / match_cnt * 100

        return match_cnt, correct_cnt, precision


    def print_pandas_results(self):

        print("\n ## Results of HPathces : num_kpt {:d},  outlier filter: {} at {}"\
                    .format(self.nfeatures, self.outlier_rejection, time.strftime("%m%d_%H%M%S") ))
        
        result_index = ["All", "illum", "view"]
        output = [self.logger.compute_avg(), self.logger_illum.compute_avg(), self.logger_view.compute_avg()]

        results_pd = pd.DataFrame(output, index=result_index, columns=[self.columns[2:]])

        pd.set_option("display.precision", 1)
        print(results_pd[['repeatability_score', 'MMA@3', 'MMA@5', "pred. match", "total_points"]])

        return {'repeatability_score': results_pd.loc['All'].at['repeatability_score'], 'MMA@3': results_pd.loc['All'].at['MMA@3'], \
                'MMA@5': results_pd.loc['All'].at['MMA@5'], "pred_match": results_pd.loc['All'].at['pred. match']} 


class HPatchesAverageMeter:
    def __init__(self):
        self.index = []; self.values =[]; self.total_datas = []

    def log_data(self, logging_data):
        seq_name, tar_idx, total_points, repeatability_score, match_cnt, correct_cnt, precision = logging_data

        self.index.append(seq_name)
        self.values.append([1, tar_idx, total_points, repeatability_score, match_cnt, \
                    correct_cnt[0],  precision[0], correct_cnt[2],  precision[2], correct_cnt[4],  precision[4]])
        self.total_datas.append([total_points, repeatability_score, match_cnt, \
                    correct_cnt[0],  precision[0], correct_cnt[2],  precision[2], correct_cnt[4],  precision[4]])

    def compute_avg(self):

        data_mean = np.mean(np.array(self.total_datas), axis=0).tolist()

        return data_mean

