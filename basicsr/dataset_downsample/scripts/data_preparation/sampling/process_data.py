

import os
import sys
sys.path.append('/data/sjq/GPU_Parallel/day_2')
from scripts.data_preparation.extract_subimages import extract_subimages

from upsample import upsample

from sklearn.model_selection import train_test_split
import shutil

def train_test(folder_path):


    # 获取文件夹中所有图片的文件名
    all_images = [file for file in os.listdir(folder_path) if file.endswith('.jpg') or file.endswith('.png')]
    # 设置随机种子以保证可重复性
    random_seed = 42

    # 随机划分为训练集和测试集，这里将测试集占比设置为 20%
    train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=random_seed)

    # 创建保存训练集和测试集的文件夹
    train_folder = os.path.join(folder_path, 'train')
    test_folder = os.path.join(folder_path, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 将图片移动到相应的文件夹
    for image in train_images:
        shutil.move(os.path.join(folder_path, image), os.path.join(train_folder, image))

    for image in test_images:
        shutil.move(os.path.join(folder_path, image), os.path.join(test_folder, image))



def process_single_folder(input_folder,save_folder,crop_size):
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3
    # HR images

    opt['input_folder'] = input_folder
    opt['save_folder'] = save_folder
    opt['crop_size'] = crop_size
    opt['step'] = int(crop_size/2)
    opt['thresh_size'] = 0
    extract_subimages(opt)

def main(args):



    input_folder = args.input_folder
    jpeg_folder = args.jpeg_folder
    output_folder = args.output_folder 
    
    up_scale = args.up_scale
    sub_size = args.sub_size

    if args.use_split:
        split_folder = args.split_folder
        train_test(split_folder)

    if args.use_jpeg:
        mod_scale = args.mod_scale
        up = upsample(up_scale,mod_scale,input_folder,jpeg_folder)
        up.generate_bicubic_img()


    if args.use_sub:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        process_single_dataset(jpeg_folder,output_folder,sub_size,up_scale)
    

    



def process_single_dataset(input_folder,output_folder,crop_size,down_factor):



    input_folder1 = os.path.join(input_folder, 'MOD')
    save_folder1 = os.path.join(output_folder, 'HR')


    process_single_folder(input_folder1,save_folder1,crop_size)



    input_folder2 = os.path.join(input_folder, 'LR')
    save_folder2 = os.path.join(output_folder, 'LR')


    process_single_folder(input_folder2,save_folder2,int(crop_size/down_factor))


    input_folder3 = os.path.join(input_folder, 'SR')
    save_folder3 = os.path.join(output_folder, 'SR')


    process_single_folder(input_folder3,save_folder3,crop_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='split-dense-sub_picture')
    # Split
    parser.add_argument('--use_split', default = False,
        help='name of the data folder')
    parser.add_argument('--split_folder', default = '/data/sjq/GPU_Parallel/day_2/datasets/Set5/ORI',
        help='name of the data folder')
    

    #Jpeg
    parser.add_argument('--use_jpeg', default = True,
        help='name of the data folder')
    parser.add_argument('--up_scale', default = 8,
        help='name of the data folder')
    parser.add_argument('--mod_scale', default = 8,
        help='name of the data folder')
    
    parser.add_argument('--input_folder', default = '/data/sjq/srDataset/EXP_data/SD300_exp/HR',
        help='name of the data folder')
    parser.add_argument('--jpeg_folder', default = '/data/sjq/srDataset/EXP_data/SD300_exp/',
        help='name of the data folder')
 
    

    #Sub
    parser.add_argument('--use_sub', default = False,
        help='name of the data folder')
    parser.add_argument('--sub_size', default = 64,
        help='name of the data folder')
    parser.add_argument('--output_folder', default = '/data/sjq/GPU_Parallel/day_2/datasets/Set5_SUB',
        help='name of the data folder')

    
    
    
    args = parser.parse_args()
    main(args)

# python scripts/data_preparation/sampling/process_data.py