import os, cv2, tqdm
import numpy as np
import data.dataset_utils as tools
from torch.utils.data import Dataset
from evaluation.evaluation_tools import check_directory

## For new data generation.
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class pytorch_dataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data =data

        ## Restrict the number of training and validation examples 
        if mode == 'train':
            if len(self.data) > 15000:
                self.data = self.data[:15000]
        elif mode == 'val':
            if len(self.data) > 3000:
                self.data = self.data[:3000]

        print('mode : {} the number of examples : {}'.format(mode, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src, scale_src_2_dst, angle_src_2_dst = self.data[idx]

        return im_src_patch[0], im_dst_patch[0], homography_src_2_dst[0], homography_dst_2_src[0], scale_src_2_dst[0], angle_src_2_dst[0]


class DatasetGeneration(object):
    def __init__(self, args):
        self.size_patches = args.patch_size
        self.data_dir = args.data_dir

        self.max_angle = args.max_angle
        self.min_scaling = args.min_scale
        self.max_scaling = args.max_scale
        self.max_shearing = args.max_shearing
        self.num_training = args.num_training_data
        self.is_debugging = args.is_debugging

        self.synth_dir = args.synth_dir
        check_directory(self.synth_dir)

        ## Input lists
        self.training_data = [] ## input_image_pairs : self.input_image_pairs / self.src2dst_Hs / self.dst2src_Hs / self.angles
        self.validation_data = []

        if not self._init_dataset_path():
            self.images_info = self._load_data_names(self.data_dir)

            self._create_synthetic_pairs(is_val=False)
            self._create_synthetic_pairs(is_val=True)
        else:
            self._load_synthetic_pairs(is_val=False)
            self._load_synthetic_pairs(is_val=True)

        print("# of Training / validation : ", len(self.training_data), len(self.validation_data))


    def get_training_data(self):
        return self.training_data

    def get_validation_data(self):
        return self.validation_data

    def _init_dataset_path(self):
        if self.is_debugging:
            self.save_path = os.path.join(self.synth_dir ,'train_dataset_debug')
            self.save_val_path = os.path.join(self.synth_dir , 'val_dataset_debug')
        else:
            self.save_path = os.path.join(self.synth_dir , 'train_dataset')
            self.save_val_path = os.path.join(self.synth_dir , 'val_dataset')

        is_dataset_exists = os.path.exists(self.save_path) and os.path.exists(self.save_val_path)

        return is_dataset_exists

    def _load_data_names(self, data_dir):
        assert os.path.isdir(data_dir), "Invalid directory: {}".format(data_dir)

        count = 0
        images_info = []

        for r, d, f in os.walk(data_dir):
            for file_name in f:
                if file_name.endswith(".JPEG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    # images_info.append(os.path.join(data_dir, r, file_name))
                    images_info.append(os.path.join(r, file_name))
                    count += 1

        src_idx = np.random.permutation(len(np.asarray(images_info)))
        images_info = np.asarray(images_info)[src_idx]

        print("Total images in directory at \" {} \" is : {}".format(data_dir, len(images_info)))

        return images_info

    def _create_synthetic_pairs(self, is_val):
        print('Generating Synthetic pairs . . . ' )

        paths = self._make_dataset_dir(is_val)

        # More stable repeatability when using bigger size patches
        if is_val:
            size_patches = 2 * self.size_patches
            self.counter += 1
        else:
            size_patches = self.size_patches
            self.counter = 0

        counter_patches = 0

        iterate = tqdm.tqdm(range(len(self.images_info)), total=len(self.images_info), desc="REKD dataset generation")

        for path_image_idx in iterate:
            name_image_path = self.images_info[(self.counter+path_image_idx) % len(self.images_info)]

            correct_patch = False

            for _ in range(10): ## looping 10 cases

                src_c = cv2.imread(name_image_path)

                minimum_length = ((size_patches ** 2) /2) ** (1/2) * 2
                if src_c.shape[0] < minimum_length or src_c.shape[1] < minimum_length:
                    continue

                hom, scale, angle, _ = tools.generate_composed_homography(self.max_angle, self.min_scaling, self.max_scaling, self.max_shearing)
                
                src, dst, dst_c = self._generate_pair(src_c, scale, angle, size_patches)

                if self._is_correct_size(src, dst, size_patches):
                    continue

                if not self._is_enough_edge(src, dst):
                    continue

                correct_patch = True
                break

            if correct_patch:
                im_src_patch = src.reshape((1, src.shape[2], src.shape[0], src.shape[1]))
                im_dst_patch = dst.reshape((1, dst.shape[2], dst.shape[0], dst.shape[1]))

                homography = self._generate_homography(src, hom, size_patches)
                homography_dst_2_src = self._preprocess_homography(homography)
                homography_src_2_dst = self._preprocess_homography(np.linalg.inv(homography))

                angle_src_2_dst = np.array([angle]).reshape(1,1)  
                scale_src_2_dst = np.array([1/scale]).reshape(1,1) 

                data = [im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src, scale_src_2_dst, angle_src_2_dst]
                self._update_data(data, is_val)

                self._save_synthetic_pair(paths, data, name_image_path.split('/')[-1])

                # visualize_synthetic_pair(src_c, dst_c, im_src_patch, im_dst_patch, homography, size_patches, angle, scale)

                counter_patches += 1

                iterate.set_description("Generated pairs : {} {:.2f}".format(counter_patches, angle) )

            ## Select the number of training patches and validation patches (and debug mode). 
            if is_val and counter_patches > 100:
                break
            elif counter_patches > self.num_training:
                break
            if is_val and self.is_debugging and counter_patches > 10:
                break
            elif not is_val and self.is_debugging and counter_patches > 400:
                break

        self.counter = counter_patches

    def _make_dataset_dir(self, is_val):

        save_path = self.save_val_path if is_val else self.save_path

        check_directory('datasets')
        check_directory(save_path)

        path_im_src_patch = os.path.join(save_path, 'im_src_patch')
        path_im_dst_patch = os.path.join(save_path, 'im_dst_patch')
        path_homography_src_2_dst = os.path.join(save_path, 'homography_src_2_dst')
        path_homography_dst_2_src = os.path.join(save_path, 'homography_dst_2_src')
        path_scale_src_2_dst = os.path.join(save_path, 'scale_src_2_dst')
        path_angle_src_2_dst = os.path.join(save_path, 'angle_src_2_dst')

        check_directory(path_im_src_patch)
        check_directory(path_im_dst_patch)
        check_directory(path_homography_src_2_dst)
        check_directory(path_homography_dst_2_src)
        check_directory(path_scale_src_2_dst)
        check_directory(path_angle_src_2_dst)

        return path_im_src_patch, path_im_dst_patch, path_homography_src_2_dst, path_homography_dst_2_src, path_scale_src_2_dst, path_angle_src_2_dst


    def _generate_pair(self, src_c, scale, angle, size_patches):
        H_resize, W_resize = round(src_c.shape[0] / scale), round(src_c.shape[1] / scale)

        image = transforms.ToTensor()(src_c)

        rotated_image = TF.rotate(image, angle)  
        rotated_scaled_image = TF.resize(rotated_image, size=(H_resize, W_resize)) 
        rotated_scaled_image = TF.center_crop(rotated_scaled_image, size_patches)

        src_image = TF.center_crop(image, size_patches)

        src_image = np.asarray(TF.to_pil_image(src_image))
        dst_c = np.asarray(TF.to_pil_image(rotated_scaled_image))

        src = tools.to_black_and_white(src_image)
        dst = tools.color_distorsion(dst_c)

        return src, dst, dst_c

    def _is_correct_size(self, src, dst, size_patches):
        return src.shape[0] != size_patches or src.shape[1] != size_patches or \
        dst.shape[0] != size_patches or dst.shape[1] != size_patches


    def _is_enough_edge(self,src,dst): 
        ## (pre-processing) If src/dst image does not have "enough edge", then continue. (Sobel edge filter)
        src_sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        src_sobelx = abs(src_sobelx.reshape((src.shape[0], src.shape[1], 1)))
        src_sobelx = src_sobelx.astype(float) / src_sobelx.max()
        dst_sobelx = cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=3)
        dst_sobelx = abs(dst_sobelx.reshape((dst.shape[0], dst.shape[1], 1)))
        dst_sobelx = dst_sobelx.astype(float) / dst_sobelx.max()

        src = src.astype(float) / src.max()
        dst = dst.astype(float) / dst.max()

        label_dst_patch = dst_sobelx
        label_src_patch = src_sobelx

        is_enough_edge = not (label_dst_patch.max() < 0.25 or label_src_patch.max() < 0.25)
        return is_enough_edge


    def _generate_homography(self, src, hom, size_patches):
        ## For GT-homography generation in image space
        inv_h = np.linalg.inv(hom)
        inv_h = inv_h / inv_h[2, 2]

        window_point = [src.shape[0]/2, src.shape[1]/2]  ## Input Image shape center point.
        point_src = [window_point[0], window_point[1], 1.0]
        point_dst = inv_h.dot([point_src[1], point_src[0], 1.0])
        point_dst = [point_dst[1] / point_dst[2], point_dst[0] / point_dst[2]]

        h_src_translation = np.asanyarray([[1., 0., -(int(point_src[1]) - size_patches / 2)],
                                        [0., 1., -(int(point_src[0]) - size_patches / 2)],
                                        [0., 0., 1.]])
        h_dst_translation = np.asanyarray([[1., 0., int(point_dst[1] - size_patches / 2)],
                                        [0., 1., int(point_dst[0] - size_patches / 2)],
                                        [0., 0., 1.]])

        homography = np.dot(h_src_translation, np.dot(hom, h_dst_translation))
        return homography


    def _preprocess_homography(self, h):
        h = h.astype('float32')
        h = h.flatten()
        h = h / h[8]
        h = h[:8]
        h = h.reshape((1, h.shape[0]))
        return h

    def _update_data(self, data, is_val):
        if is_val:
            self.validation_data.append(data)
        else:
            self.training_data.append(data)

    def _save_synthetic_pair(self, paths, data, name_image):
        ## Save the patches by np format (For caching)
        path_im_src_patch, path_im_dst_patch, path_homography_src_2_dst, path_homography_dst_2_src, path_scale_src_2_dst, path_angle_src_2_dst = paths
        im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src, scale_src_2_dst, angle_src_2_dst =  data       

        np.save(os.path.join(path_im_src_patch, name_image), im_src_patch)
        np.save(os.path.join(path_im_dst_patch, name_image), im_dst_patch)
        np.save(os.path.join(path_homography_src_2_dst, name_image), homography_src_2_dst)
        np.save(os.path.join(path_homography_dst_2_src, name_image), homography_dst_2_src)
        np.save(os.path.join(path_scale_src_2_dst, name_image), scale_src_2_dst)
        np.save(os.path.join(path_angle_src_2_dst, name_image), angle_src_2_dst)

    def _load_synthetic_pairs(self, is_val):
        print('Loading Synthetic pairs . . .')

        path_im_src_patch, path_im_dst_patch, path_homography_src_2_dst, path_homography_dst_2_src, \
                        path_scale_src_2_dst, path_angle_src_2_dst = self._make_dataset_dir(is_val)

        for name_image in tqdm.tqdm(os.listdir(path_im_src_patch), total=len(os.listdir(path_im_src_patch))):
            if name_image[-8:] != "JPEG.npy":
                continue
            ## Load the patches by np format (caching)
            im_src_patch = np.load(os.path.join(path_im_src_patch, name_image))
            im_dst_patch = np.load(os.path.join(path_im_dst_patch, name_image))
            homography_src_2_dst = np.load(os.path.join(path_homography_src_2_dst, name_image))
            homography_dst_2_src = np.load(os.path.join(path_homography_dst_2_src, name_image))
            scale_src_2_dst = np.load(os.path.join(path_scale_src_2_dst, name_image))
            angle_src_2_dst = np.load(os.path.join(path_angle_src_2_dst, name_image))

            data = [im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src, scale_src_2_dst, angle_src_2_dst]

            self._update_data(data, is_val)
            


#### ===========  for input image sanity check ============= ### 

def visualize_synthetic_pair(src_c, dst_c, im_src_patch, im_dst_patch, homography, size_patches, angle, scale):
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from kornia.geometry.transform import warp_perspective

    def _reconstruct_images(dst_c, scale, angle, size_patches, im_dst_patch, homography):
        ## recon 1 : using sca/ori
        H_recon, W_recon = round(dst_c.shape[0]*scale), round(dst_c.shape[1] * scale)  ## This is for TF.resize() function.
        dst_src_recon = transforms.ToTensor()(dst_c)
        dst_src_recon = TF.resize(dst_src_recon, size=(H_recon, W_recon)) ## scale
        dst_src_recon = TF.rotate(dst_src_recon, -angle)  ## angle
        dst_src_recon = TF.center_crop(dst_src_recon, size_patches)
        dst_src_recon = np.asarray(TF.to_pil_image(dst_src_recon))

        ## recon 2 : using homography
        dst_src_recon1 = warp_perspective(torch.tensor(im_dst_patch).to(torch.float32), torch.tensor(homography.astype('float32')).unsqueeze(0),  dsize=(size_patches, size_patches)).squeeze(0)
        dst_src_recon1 = np.asarray(TF.to_pil_image(dst_src_recon1.to(torch.uint8)))

        return dst_src_recon, dst_src_recon1

    dst_src_recon, dst_src_recon1 = _reconstruct_images(dst_c, scale, angle, size_patches, im_dst_patch, homography)
    src_dst_recon, src_dst_recon1 = _reconstruct_images(src_c, 1/scale, -angle, size_patches, im_src_patch, np.linalg.inv(homography))

    fig = plt.figure(figsize=(15,8))
    rows = 2 ; cols = 5
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(src_c)
    ax1.set_title('src_c (input image)')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(im_src_patch[0,0,:,:],  cmap='gray')
    ax2.set_title('im_src_patch')
    ax2.axis("off")

    ax3 = fig.add_subplot(rows, cols, 7)
    ax3.imshow(im_dst_patch[0,0,:,:],  cmap='gray')
    ax3.set_title('im_dst_patch')
    ax3.axis("off")

    ax4 = fig.add_subplot(rows, cols, 4)
    ax4.imshow(dst_src_recon[:,:],  cmap='gray')
    ax4.set_title('im_dst_to_src(sca/ori)')
    ax4.axis("off")

    ax5 = fig.add_subplot(rows, cols, 5)
    ax5.imshow(dst_src_recon1[:,:],  cmap='gray')
    ax5.set_title('im_dst_to_src(hom)')
    ax5.axis("off")


    ax6 = fig.add_subplot(rows, cols, 6)
    ax6.imshow(dst_c,  cmap='gray')
    ax6.set_title('dst_c (input image)')
    ax6.axis("off")

    ax7 = fig.add_subplot(rows, cols, 9)
    ax7.imshow(src_dst_recon[:,:],  cmap='gray')
    ax7.set_title('im_src_to_dst(sca/ori)')
    ax7.axis("off")

    ax8 = fig.add_subplot(rows, cols, 10)
    ax8.imshow(src_dst_recon1[:,:],  cmap='gray')
    ax8.set_title('im_src_to_dst(hom)')
    ax8.axis("off")

    plt.show()
