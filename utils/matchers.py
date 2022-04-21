import cv2, torch
import numpy as np


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    descriptors_a= descriptors_a.to(device=device)
    descriptors_b= descriptors_b.to(device=device)
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def orb_matcher(d1,d2):
    norm = cv2.NORM_HAMMING
    matcher = cv2.BFMatcher_create(norm, crossCheck=True)
    matches_cv = matcher.match(d1, d2)
    matches = []
    for match in matches_cv:
        matches.append([match.queryIdx, match.trainIdx])
    matches = np.array(matches)
    return matches 

class OpencvBruteForceMatcher(object):
     name = 'opencv_brute_force_matcher'
     distances = {}
     distances['l2'] = cv2.NORM_L2
     distances['hamming'] = cv2.NORM_HAMMING

     def __init__(self, distance='l2'):
         self._matcher = cv2.BFMatcher(self.distances[distance])

     def match(self, descs1, descs2):
         """Compute brute force matches between two sets of descriptors.
         """
         assert isinstance(descs1, np.ndarray), type(descs1)
         assert isinstance(descs2, np.ndarray), type(descs2)
         assert len(descs1.shape) == 2, descs1.shape
         assert len(descs2.shape) == 2, descs2.shape
         matches = self._matcher.match(descs1, descs2)
         return matches

     def match_putative(self, descs1, descs2, knn=2, threshold_ratio=0.7):
         """Compute putatives matches betweem two sets of descriptors.
         """
         assert isinstance(descs1, np.ndarray), type(descs1)
         assert isinstance(descs2, np.ndarray), type(descs2)
         assert len(descs1.shape) == 2, descs1.shape
         assert len(descs2.shape) == 2, descs2.shape
         matches = self._matcher.knnMatch(descs1, descs2, k=knn)
         # apply Lowe's ratio test
         good = []
         for m, n in matches:
             if m.distance < threshold_ratio * n.distance:
                 good.append(m)
         return good

     def convert_opencv_matches_to_numpy(self, matches):
         """Returns a np.ndarray array with points indices correspondences
            with the shape of Nx2 which each N feature is a vector containing
            the keypoints id [id_ref, id_dst].
         """
         assert isinstance(matches, list), type(matches)
         correspondences = []
         for match in matches:
             assert isinstance(match, cv2.DMatch), type(match)
             correspondences.append([match.queryIdx, match.trainIdx])
         return np.asarray(correspondences)
         
