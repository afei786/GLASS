import cv2
import os
from tqdm import tqdm
import numpy as np
def resize(img, size):
    img = cv2.resize(img, size)
    
    return img


def resize_with_padding(image, target_size):
    """
    等比例缩放图片并填充到目标尺寸。
    
    Args:
        image (np.array): 输入图片。
        target_size (tuple): 目标尺寸 (height, width)。
        
    Returns:
        resized_image (np.array): 处理后的图片。
    """
    original_height, original_width = image.shape[:2]
    target_height, target_width = target_size

    # 计算缩放比例
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 等比例缩放图片
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建目标尺寸的黑色背景
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 将缩放后的图片放置在背景中央
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image

def main():
    img_path = '/home/fei/code/jm_wh3/images_seg/fql'
    new_path = '/home/fei/data/wuhu/images_seg/fql'
    os.makedirs(new_path, exist_ok=True)
    img_list = os.listdir(img_path)
    for img_name in tqdm(img_list):
        img = cv2.imread(os.path.join(img_path, img_name))
        # print(img.shape)
        # img = resize(img, (640, 640))
        # cv2.imwrite(os.path.join(new_path, img_name), img)
    
        target_size = (640, 640)  # 目标尺寸
        result = resize_with_padding(img, target_size)

        cv2.imwrite(os.path.join(new_path, img_name), result)    



def main2():
    img_path = '/home/fei/code/jm_wh3/new_images2_seg2'
    new_path = '/home/fei/data/wuhu/new_images2_seg2_resize640'
    os.makedirs(new_path, exist_ok=True)
    class_name = os.listdir(img_path)
    for class_name_ in tqdm(class_name):
        img_list = os.listdir(os.path.join(img_path, class_name_))
        new_img_path = os.path.join(new_path, class_name_)
        os.makedirs(new_img_path, exist_ok=True)
        for img_name in tqdm(img_list):
            img = cv2.imread(os.path.join(img_path, class_name_, img_name))
            # print(img.shape)
            # img = resize(img, (640, 640))

            target_size = (640, 640)  # 目标尺寸
            result = resize_with_padding(img, target_size)
            cv2.imwrite(os.path.join(new_img_path, img_name), result)



if __name__ == '__main__':
    main2()