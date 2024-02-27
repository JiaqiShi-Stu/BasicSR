import cv2
import os
import sys
sys.path.append('/data/sjq/GPU_Parallel/day_2')
import math
import random
import numpy as np
import os.path as osp
from scipy.io import loadmat
from PIL import Image
import torch

from Utils_DATA import gaussian_kernels as gaussian_kernels

from basicsr.data.transforms import augment







class upsample():
    def __init__(self,upscale,modscale,input_folder,save_folder):



        self.use_motion_kernel = False


        if self.use_motion_kernel:
            self.motion_kernel_prob = 0.1
            motion_kernel_path = 'basicsr/data/motion-blur-kernels-32.pth'
            self.motion_kernels = torch.load(motion_kernel_path)


        '''StableSR'''
        # blur_kernel_size: 21
        # kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        # kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        # sinc_prob: 0.1
        # blur_sigma: [0.2, 1.5]
        # betag_range: [0.5, 2.0]
        # betap_range: [1, 1.5]

        # blur_kernel_size2: 11
        # kernel_list2: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        # kernel_prob2: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        # sinc_prob2: 0.1
        # blur_sigma2: [0.2, 1.0]
        # betag_range2: [0.5, 2.0]
        # betap_range2: [1, 1.5]
            
        '''codeformer'''
        # in_size: 512
        # gt_size: 512
        # mean: [0.5, 0.5, 0.5]
        # std: [0.5, 0.5, 0.5]
        # use_hflip: true
        # use_corrupt: true

        # blur_kernel_size: 41
        # use_motion_kernel: false
        # motion_kernel_prob: 0.001
        # kernel_list: ['iso', 'aniso']
        # kernel_prob: [0.5, 0.5]
        # # small degradation in stageIII
        # blur_sigma: [0.1, 10]
        # downsample_range: [1, 6]
        # noise_range: [0, 15]
        # jpeg_range: [60, 100]
        # # large degradation in stageII
        # blur_sigma_large: [1, 10]
        # downsample_range_large: [4, 8]
        # noise_range_large: [0, 20]
        # jpeg_range_large: [30, 80]

        self.blur_kernel_size = 21
        self.blur_sigma = [0.2,1.5]
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5,0.5]
        self.downsample_range = [4,4]

        self.noise_range =  [0, 20]
        self.jpeg_range =  [30, 80]
        self.use_hflip = False
        self.use_corrupt = True


        



        self.modscale = modscale
        self.input_folder = input_folder
        self.save_folder = save_folder




    def generate_bicubic_img(self):
        
        save_mod_folder = os.path.join(self.save_folder, 'MOD')
        save_lr_folder = os.path.join(self.save_folder, 'LR')
        save_sr_folder = os.path.join(self.save_folder, 'SR')

        if not os.path.exists(save_mod_folder):
            os.makedirs(save_mod_folder)
        if not os.path.exists(save_lr_folder):
            os.makedirs(save_lr_folder)
        if not os.path.exists(save_sr_folder):
            os.makedirs(save_sr_folder)

        idx = 0
        filepaths = [f for f in os.listdir(self.input_folder) if os.path.isfile(os.path.join(self.input_folder, f))]
        for filepath in filepaths:
            img_name, _ = os.path.splitext(filepath)
            
            img = cv2.imread(os.path.join(self.input_folder, filepath))
            idx += 1
            print(f'{idx}\t{img_name}')
            img = img.astype(float) / 255.0

            # modcrop
            img = self.modcrop(img)
            h, w = img.shape[:2]
            cv2.imwrite(os.path.join(save_mod_folder, img_name + '.png'), img * 255.0)
            


            # random horizontal flip
            img_gt, status = augment(img, hflip=self.use_hflip, rotation=False, return_status=True)


            # generate in image
            img_in = img_gt
            if self.use_corrupt:
                # motion blur
                if self.use_motion_kernel and random.random() < self.motion_kernel_prob:
                    m_i = random.randint(0,31)
                    k = self.motion_kernels[f'{m_i:02d}']
                    img_in = cv2.filter2D(img_in,-1,k)
                
                # gaussian blur
                kernel = gaussian_kernels.random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    self.blur_kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, 
                    [-math.pi, math.pi],
                    noise_range=None)
                img_in = cv2.filter2D(img_in, -1, kernel)

                # downsample
                scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
                img_in = cv2.resize(img_in, (int(h // scale), int(w // scale)), interpolation=cv2.INTER_LINEAR)

                # noise
                if self.noise_range is not None:
                    noise_sigma = np.random.uniform(self.noise_range[0] / 255., self.noise_range[1] / 255.)
                    noise = np.float32(np.random.randn(*(img_in.shape))) * noise_sigma
                    img_in = img_in + noise
                    img_in = np.clip(img_in, 0, 1)

                # jpeg

                if self.jpeg_range is not None:
                    jpeg_p = np.random.uniform(self.jpeg_range[0], self.jpeg_range[1])
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_p)]
                    _, encimg = cv2.imencode('.jpg', (img_in * 255).astype(np.uint8), encode_param)
                    img_in = np.float32(cv2.imdecode(encimg, 1)) / 255.


                #LR
                img_in = img_in * 255.0
                cv2.imwrite(os.path.join(save_lr_folder, img_name + '.png'), img_in)


                


                # resize to in_size
                img_in = cv2.resize(img_in, (h, w), interpolation=cv2.INTER_LINEAR)
                # SR
                cv2.imwrite(os.path.join(save_sr_folder, img_name + '.png'), img_in)


    def modcrop(self,img):
        modulo = self.modscale
        h, w = img.shape[:2]
        h = h - h % modulo
        w = w - w % modulo
        return img[:h, :w]




# 调用函数
if __name__ == "__main__":

    input_folder = r'/data/sjq/GPU_Parallel/day_2/datasets/Set5/ORI'
    save_folder = r'/data/sjq/GPU_Parallel/day_2/datasets/Set5/SUB'
    up_scale = 8
    mod_scale = 8



    up = upsample(up_scale,mod_scale,input_folder,save_folder)
    up.generate_bicubic_img()



