import skimage.transform as trans
import SimpleITK as sitk
import numpy as np
import os

from pathlib import Path

def save_sitk(input, name):
    '''
    input: array or image 数组的话自动去掉大小为1的维度
    name: 仅通过'.'判断是否需要自己加后缀
    '''
    if not '.' in name:
        name = f'{name}.nii'
    if isinstance(input, np.ndarray):
        sitk.WriteImage(sitk.GetImageFromArray(input.squeeze()), name)
    else: sitk.WriteImage(input, name)


def get_all_files(directory): 
    # return : 都是文件,不包含文件夹。 哪怕这个文件夹是空的

    all_files = []
    
    # 使用 os.walk() 遍历目录
    for root, dirs, files in os.walk(directory):
        # 遍历当前目录下的所有文件
        for file in files:
            # 拼接完整的文件路径
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    
    return all_files


def filter_image(name_list, contain, non_contain):
    # newlist = filter(lambda x: x % 2 == 0, name_list)
    filtered_files1 = list(filter(
        lambda x: any(x.endswith(included) for included in contain),
        name_list
    ))
    filtered_files2 = list(filter(
        lambda x: not any(x.endswith(excluded) for excluded in non_contain),
        filtered_files1
    ))
    return filtered_files2


def init_mean_std(output_dir, data_loader):
    if os.path.isfile(os.path.join(output_dir, 'mean_std.npy')):
        tmp = np.load(os.path.join(output_dir, 'mean_std.npy'))
        mean_vector, std_vector = tmp
    else:
        mean_vector, std_vector = None, None
        print('Calculating mean and std')
        for data in data_loader:
            image = data[1]
            if mean_vector is None:
                mean_vector = np.zeros(image.size(1))
                std_vector = np.zeros(image.size(1))
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += image[:, j, :, :].mean()
                std_vector[j] += image[:, j, :, :].std()

        mean_vector /= (len(data_loader))
        std_vector /= (len(data_loader))
        mean_std_fullname = os.path.join(output_dir, f'mean_std.npy')
        print(f'Save mean_std to {os.path.abspath(mean_std_fullname)}')
        np.save(os.path.join(output_dir, f'mean_std.npy'),
                [mean_vector.astype(np.float32), std_vector.astype(np.float32)])

    return mean_vector, std_vector


def read_list(txt_name):
    with open(txt_name, "r") as f:
        train_list = f.read()
    return train_list.split('\n')


def resample_3D_nii_to_Fixed_size(nii_image, image_new_size, resample_methold=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    image_original_size = nii_image.GetSize()  # 原始图像的尺寸
    image_original_spacing = nii_image.GetSpacing()  # 原始图像的像素之间的距离
    image_new_size = np.array(image_new_size, float)
    factor = image_original_size / image_new_size
    image_new_spacing = image_original_spacing * factor
    image_new_size = image_new_size.astype(np.int)

    resampler.SetReferenceImage(nii_image)  # 需要resize的图像（原始图像）
    resampler.SetSize(image_new_size.tolist())
    resampler.SetOutputSpacing(image_new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_methold)

    return resampler.Execute(nii_image)


def nii_resize_2D(image, label, shape):
    """
    type of image,label: Image or array or None

    :return: array or None
    """
    # image
    if isinstance(image, sitk.SimpleITK.Image):  # image need type array, if not, transform it
        image = sitk.GetArrayFromImage(image)
    if image is not None:
        image = trans.resize(image, (shape, shape))
    # label
    if isinstance(label, np.ndarray):
        label = sitk.GetImageFromArray(label)  # label1 need type Image
    if label is not None:
        label = resample_3D_nii_to_Fixed_size(label, (shape, shape),
                                              resample_methold=sitk.sitkNearestNeighbor)
        label = sitk.GetArrayFromImage(label)
    return image, label
