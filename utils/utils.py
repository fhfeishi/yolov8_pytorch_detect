import random
import cv2
import numpy as np
import torch
from PIL import Image

import yaml
def load_cfg():
    with open("cfg.yaml", "r", encoding='utf-8') as file:
        return yaml.safe_load(file)

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def resize_myimage(image, size, letterbox_image):
    w, h = size
    if isinstance(image, Image.Image):
        iw, ih = image.size
        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

    elif isinstance(image, np.ndarray):
        ih, iw = image.shape[:2]
        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.full((h, w, 3), 128, dtype=np.uint8)
            new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = resized_image
        else:
            new_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        return new_image
    else:
        raise TypeError("输入图像数据非常规opencv Img or PIL Image")

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


# show opencv image
def show_cvimage(image_data, mode=0, title="color", w=1280, h=720):
    if image_data is None:
        raise ValueError("Input image data is None!")

    assert mode in [0, 1], "Mode must be 0 (static image) or 1 (video stream)!"

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, w, h)
    cv2.imshow(title, image_data)

    if mode == 0:
        # 显示静态图像，等待用户按键
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode == 1:
        # 显示视频流帧，短暂等待以刷新界面
        key = cv2.waitKey(1) & 0xFF
        # 可选：处理按键事件，例如按下 'q' 键退出
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False  # 返回 False 以指示停止处理
    return True  # 返回 True 继续处理
