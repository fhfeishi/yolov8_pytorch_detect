import colorsys, os, PIL, cv2, av, time, torch,random 
import numpy as np
import torch.nn as nn
from PIL import ImageDraw, ImageFont, ImageOps
from nets.yolo_for_train import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_myimage, show_config, load_cfg)
from utils.utils_bbox import DecodeBox, check_version
from collections import OrderedDict, defaultdict
import xml.etree.ElementTree as ET
import xml.dom.minidom

# read config.yaml
cfg = load_cfg()

class SlideImage(object):
    def __init__(self):
        self.w = None
        self.h = None
        self.scaled_h = None
        self.scaled_w = None
        self.crop_size_w = None
        self.crop_size_h = None
        self.stride_w = None
        self.stride_h = None
        self.padding_w = None
        self.padding_h = None
        self.padding_bottom = None
        self.padding_right = None
        self.padding_top = None
        self.padding_left = None
        self.cols_num = None  
        self.rows_num = None   
        self._ImageBlock = {}  # key: (row_index, col_index), value:image_block

    def pillow_slide(self, pillowimage, crop_size, rows_num=None, cols_num=None,
                     stride=None, fill_=128, drawBoxForTest=False):
        """
        ## 1
        pillowimage
        crop_size = (a, b)
        rows_num, cols_num = c,d

        ## 2
        pillowimage
        crop_size = (a, b)
        stride = (c,d)
        """
        self.w, self.h = pillowimage.size
        assert crop_size is not None, "crop_size is None !!!"
        self.crop_size_w, self.crop_size_h = crop_size
        
        if self.h < crop_size[1] or self.w < crop_size[0]:
            print("image is too small !!! retrun is input_pillowimage")
            return pillowimage 
        
        
        if isinstance(rows_num, int) and isinstance(cols_num, int):
            # 1
            assert cols_num > 1 and rows_num > 1, "cols_num, rows_num must > 1"
            self.rows_num, self.cols_num = rows_num, cols_num
            # no padding
            self.stride_w = (self.w - self.crop_size_w) // (self.cols_num - 1)
            self.stride_h = (self.h - self.crop_size_h) // (self.rows_num - 1)
        elif stride is not None:
            # 2
            self.stride_w, self.stride_h = stride
            
            # rows num
            rest_h = self.h - crop_size[1]
            if rest_h % self.stride_h != 0:
                num_rows = int(rest_h // self.stride_h) + 1
            else:
                num_rows = int(rest_h / self.stride_h)
            self.rows_num = num_rows + 1

            # cols num
            rest_w = self.w - crop_size[0]
            if rest_w % self.stride_w != 0:
                num_cols = int(rest_w // self.stride_w) + 1
            else:
                num_cols = int(rest_w // self.stride_w)
            self.cols_num = num_cols + 1
            
        else:
            print("wrong params input!")
            print(f"{stride=}, {rows_num=}, {cols_num=}")

        # padding height, padding width
        self.padding_h = (self.rows_num-1) * self.stride_h + crop_size[1] - self.h
        self.padding_w = (self.cols_num-1) * self.stride_w + crop_size[0] - self.w
        self.padding_h = int(self.padding_h) + (int(self.padding_h) % 2)
        self.padding_w = int(self.padding_w) + (int(self.padding_w) % 2)

        # # ---------- padding ------------ #
        # 右边和下面补零   --左上右下
        padded_image = ImageOps.expand(pillowimage, border=(
            0, 0, self.padding_w, self.padding_h), fill=(fill_, fill_, fill_))
        self.padding_left, self.padding_top, self.padding_right, self.padding_bottom = 0, 0, self.padding_w, self.padding_h
        
        # # ------------ crop -------------- #  pillow draw ++
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                start_row = r * self.stride_h
                end_row = start_row + crop_size[1]
                start_col = c * self.stride_w
                end_col = start_col + crop_size[0]
                block = padded_image.crop((start_col, start_row, end_col, end_row))
                if block.size == crop_size:
                    self._ImageBlock[(r, c)] = block
        
        if drawBoxForTest:
            draw = ImageDraw.Draw(padded_image)
            font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * pillowimage.size[1] + 0.5).astype('int32'))
            
            for thick, ((row_, col_), _)in enumerate(self._ImageBlock.items()):
                x1 = int(row_*self.stride_w)
                y1 = int(col_*self.stride_h)
                x2 = x1 + self.crop_size_w
                y2 = y1 + self.crop_size_h
                
                text = f"row:{row_} col:{col_}"
                text_size = (
                draw.textbbox((0, 0), text, font=font)[2] - draw.textbbox((0, 0), text, font=font)[0], \
                draw.textbbox((0, 0), text, font=font)[3] - draw.textbbox((0, 0), text, font=font)[1]) \
                    if check_version(PIL.__version__, '9.2.0') else draw.textsize(text, font)
                color = tuple(random.randint(0, 255) for _ in range(3))
                # rect bbox
                draw.rectangle([x1,y1,x2,y2], outline=color, width=thick)
                
                # text
                draw.rectangle([x1-1, y1-1, x1+text_size[0]+1, y1+text_size[1]+1], fill=color)
                draw.text((x1,y1), str(text), fill=(0, 0, 0), font=font)   
                
            padded_image.show()
        
        return self._ImageBlock

    def opencv_slide(self,opencvimage, crop_size, rows_num=None, cols_num=None,
                     stride=None, fill_=128, drawBoxForTest=False):
        """
        ## 1
        opencvimage
        crop_size = (a, b)
        rows_num, cols_num = c,d

        ## 2
        opencvimage
        crop_size = (a, b)
        stride = (c,d)
        """
        self.h, self.w = opencvimage.shape[0:2]
        assert crop_size is not None, "crop_size is None !!!"
        self.crop_size_w, self.crop_size_h = crop_size
        
        if self.h < crop_size[1] or self.w < crop_size[0]:
            print("image is too small !!! retrun is input_opencvimage")
            return opencvimage 
        
        if isinstance(rows_num, int) and isinstance(cols_num, int):
            # 1
            assert cols_num > 1 and rows_num > 1, "cols_num, rows_num must > 1"
            self.rows_num, self.cols_num = rows_num, cols_num
            # no padding
            self.stride_w = (self.w - self.crop_size_w) // (self.cols_num - 1)
            self.stride_h = (self.h - self.crop_size_h) // (self.rows_num - 1)
        elif stride is not None:
            # 2
            self.stride_w, self.stride_h = stride
            # rows num
            rest_h = self.h - crop_size[1]
            if rest_h % self.scaled_hh != 0:
                num_rows = int(rest_h // self.stride_h) + 1
            else:
                num_rows = int(rest_h / self.stride_h)
            self.rows_num = num_rows + 1

            # cols num
            rest_w = self.w - crop_size[0]
            if rest_w % self.stride_w != 0:
                num_cols = int(rest_w // self.stride_w) + 1
            else:
                num_cols = int(rest_w // self.stride_w)
            self.cols_num = num_cols + 1
            
        else:
            print("wrong params input!")
            print(f"{stride=}, {rows_num=}, {cols_num=}")

        # padding height, padding width
        self.padding_h = (self.rows_num-1) * self.stride_h + self.crop_size_h - self.h
        self.padding_w = (self.cols_num-1)* self.stride_w + self.crop_size_w - self.w
        self.padding_h = int(self.padding_h) + (int(self.padding_h) % 2)
        self.padding_w = int(self.padding_w) + (int(self.padding_w) % 2)
        

        # # ---------- padding ------------ #
        # 右边和下面补零  上下左右
        padded_image = cv2.copyMakeBorder(opencvimage, 0, self.padding_h, 0, self.padding_w,
                                          cv2.BORDER_CONSTANT, value=(fill_, fill_, fill_))

        # # ------------ crop -------------- #
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                y1 = r * self.stride_h
                y2 = y1 + self.crop_size_h
                x1 = c * self.stride_w
                x2 = x1 + self.crop_size_w 
                block = padded_image[y1:y2, x1:x2]
                if block.shape[1] == crop_size[0] and block.shape[0] == crop_size[1]:
                    self._ImageBlock[(r, c)] = block

        # print(f"{self.rows_num=} {self.cols_num=} {self.stride_w=}{self.stride_h=}{self.padding_w=}")
        
        # -----------   show crop ----------------- # 
        if drawBoxForTest:
            draw_ = padded_image.copy()
            for i, ((row, col), _) in enumerate(self._ImageBlock.items()):
                # 2x2: print(f"{row=} {col=}") # 00 01 10 11
                # Calculate coordinates
                y1 = row * self.stride_h         
                y2 = y1 + self.crop_size_h   
                x1 = col * self.stride_w        
                x2 = x1 + self.crop_size_w   
                
                thick = i+1
                color = tuple(random.randint(0,255) for _ in range(3))
                cv2.rectangle(draw_, (x1,y1), (x2, y2), color, thick)
                
                text = f"row:{row} col:{col}"
                font_scale = 2
                font_size = 3
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_size)
                text_origin = (x1, y2 + text_size[1]) # bottom-left
                # cv2.rectangle(draw_, (x1-1,y1-text_size[1]-1), (x1+text_size[0]+1, y1+1), (255,255,255), -1)
                cv2.putText(draw_, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, font_size, lineType=cv2.LINE_AA)
            # draw_ = cv2.cvtColor(draw_, cv2.COLOR_RGB2BGR)
            show_cvimage(draw_, 0)
        
        return self._ImageBlock


# merge slide_image_input_model_out
def merge_overlapped_ltbrBbox(results_list, iou_thre=0.3):
    if results_list is None:
        return None
    class_dict = defaultdict(list)
    for label, conf, bbox in results_list:
        class_dict[label].append((conf, bbox))
    final_results = []
    for label, bboxes in class_dict.items():
        bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
        while bboxes:
            best_conf, best_bbox = bboxes.pop(0)
            merged_bbox = best_bbox.copy()
            to_merge = []
            for conf, bbox in bboxes:
                if calculate_tlbrIou(best_bbox, bbox) > iou_thre:
                    to_merge.append((conf, bbox))
            for conf, bbox in to_merge:
                merged_bbox = merge_tlbrBox(merged_bbox, bbox)
                best_conf = max(best_conf, conf)
                bboxes.remove((conf, bbox))
            final_results.append([label, best_conf, merged_bbox])
        return final_results
def calculate_tlbrIou(boxA, boxB): 
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
def merge_tlbrBox(boxA, boxB):
    y1 = min(boxA[0], boxB[0])
    x1 = min(boxA[1], boxB[1])
    y2 = max(boxA[2], boxB[2])
    x2 = max(boxA[3], boxB[3])
    return np.array([y1, x1, y2, x2])

# show opencv image
def show_cvimage(image_data, mode_=0, title="color", w=1440, h=810):

    if image_data is None:
        raise ValueError("Input image data is None!")

    assert mode_ in [0, 1], "Mode must be 0 (static image) or 1 (video stream)!"

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, w, h)
    cv2.imshow(title, image_data)

    if mode_ == 0:
        # 显示静态图像，等待用户按键
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode_ == 1:
        # 显示视频流帧，短暂等待以刷新界面
        key = cv2.waitKey(1) & 0xFF
        # 可选：处理按键事件，例如按下 'q' 键退出
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False  # 返回 False 以指示停止处理
    return True  # 返回 True 继续处理

# draw on opencv image
def rgb2bgr_(rgbcolor):
    return (rgbcolor[2], rgbcolor[1], rgbcolor[0])

def get_random_color():
    """
    返回一个随机的 RGB 值，每个通道的值为 0 到 255 的整数。
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b) 

class YOLO_INFER(object):
    _defaults = {
        "model_path"        : cfg['infer_cfg']['model_path'],
        "classes_path"      : cfg['data_cfg']['classes_file'],
        "input_shape"       : cfg['model_cfg']['input_shape'],
        "phi"               : cfg['model_cfg']['phi'],
        "confidence"        : cfg['infer_cfg']['confidence'],
        "nms_iou"           : cfg['infer_cfg']['nms_iou'],
        "letterbox_image"   : cfg['infer_cfg']['letterbox_image'],
        "cuda"              : cfg['infer_cfg']['cuda']}

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.bbox_util                      = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        # show_config(**self._defaults)

    def generate(self, onnx=False):

        self.net    = YoloBody(self.input_shape, self.num_classes, self.phi)
        
        # device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.net.load_state_dict(new_state_dict)
        
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, crop = False, count = False):

        image_shape = (image.size[1], image.size[0])
        image_data  = resize_myimage(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            if results[0] is None:
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

            # top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.8)
            
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]  # 有可能会预测错误，就不用了
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            # # image classification
            # img_crop_ = image.crop((left, top, right, bottom))
            # img_crop_ = predBboxCrop_transform(img_crop_)
            # predict_cla = class_predBboxCrop(img_crop_)
            # label_text = '{} {:.2f}'.format(predict_cla, score)
            
            # label_text = '{} {:.2f}'.format(predicted_class, score)
            label_text = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = (
                draw.textbbox((0, 0), label_text, font=font)[2] - draw.textbbox((0, 0), label_text, font=font)[0], \
                draw.textbbox((0, 0), label_text, font=font)[3] - draw.textbbox((0, 0), label_text, font=font)[1]) \
                    if check_version(PIL.__version__, '9.2.0') else draw.textsize(label_text, font)

            label = label_text.encode('utf-8')
            """
            if isinstance(label, bytes):
                label = label.decode('utf-8)
            label_sparts = label.strip().split(' ')
            if len(label_parts) >= 2:
            label, score = label_parts[0], label_parts[1]
            else:
                label = label_parts[0]
                score = '1.00'  # 默认置信度
            """
            print(label_text, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def gen_xml(self, image, image_path, xml_path_save):
        # 获取图片名称和路径
        image_name = os.path.basename(image_path)
        folder_name = os.path.basename(os.path.dirname(image_path))

        image_shape = np.array(np.shape(image)[0:2])  # (h,w)
        image_data = resize_myimage(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        detection_results = []
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou
            )
            if results[0] is None:
                return None
            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i] # no use
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image_shape[0], np.floor(bottom).astype('int32'))
            right = min(image_shape[1], np.floor(right).astype('int32'))
            detection_results.append((predicted_class, score, top, left, bottom, right))

        annotation = ET.Element("annotation")
        folder = ET.SubElement(annotation, "folder")
        folder.text = folder_name
        filename = ET.SubElement(annotation, "filename")
        filename.text = image_name
        path = ET.SubElement(annotation, "path")
        path.text = image_path
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_shape[1])
        height = ET.SubElement(size, "height")
        height.text = str(image_shape[0])
        depth = ET.SubElement(size, "depth")
        depth.text = "3"

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        
        for result in detection_results:
            label, score, top, left, bottom, right = result
            obj = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj, "name")
            name.text = label
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(left)
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(top)
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(right)
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(bottom)

        # # 保存XML文件
        # tree = ET.ElementTree(annotation)
        # tree.write(xml_path_save, encoding="utf-8", xml_declaration=True)

        # 使用minidom格式化XML并写入文件
        xml_str = ET.tostring(annotation, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)  # 解析并格式化XML
        with open(xml_path_save, 'w', encoding='utf-8') as f:
            f.write(dom.toprettyxml(indent="  "))  # 使用2个空格缩进进行格式化     
    
    def dectect_pilimage(self, image, show_=False):
        image_shape = (image.size[1], image.size[0]) 
        image_data = resize_myimage(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou
            )

            if results[0] is None:
                return image, None, None, None

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

            top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.8)

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label_text = '{} {:.2f}'.format(predicted_class, score)
            
            draw = ImageDraw.Draw(image)
            label_size = (
                draw.textbbox((0, 0), label_text, font=font)[2] - draw.textbbox((0, 0), label_text, font=font)[0], \
                draw.textbbox((0, 0), label_text, font=font)[3] - draw.textbbox((0, 0), label_text, font=font)[1]) \
                if check_version(PIL.__version__, '9.2.0') else draw.textsize(label_text, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label_text, fill=(0, 0, 0), font=font)
            del draw

        if show_:
            image.show()

        return image, top_boxes, top_conf, top_label

    def detect_cvimage(self, image, show_=False):
        image_shape = np.shape(image)[0:2]
        image_data = resize_myimage(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou)

            if results[0] is None:
                return image, None, None, None

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

            # ## nms cross class
            # top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.9)

        # 设置字体和线条粗细
        font_scale = 1
        thickness = 2

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom).astype('int32'))
            right = min(image.shape[1], np.floor(right).astype('int32'))

            label_text = '{}'.format(predicted_class)

            # 绘制预测框
            cv2.rectangle(image, (left, top), (right, bottom), rgb2bgr_(self.colors[c]), thickness)

            # 计算文本尺寸
            (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # 判断文本是否会超出图像顶部
            if top - label_height - thickness >= 0:
                # 文本在预测框上方
                text_origin = (left, top - thickness - 1)
                # 绘制文本背景
                cv2.rectangle(image, (left, top - label_height - thickness - 1),
                              (left + label_width + 2, top - thickness - 1), rgb2bgr_(self.colors[c]), -1)
            else:
                # 文本在预测框下方
                text_origin = (left, top + label_height + thickness + 1)
                # 绘制文本背景
                cv2.rectangle(image, (left, top + thickness + 1),
                              (left + label_width + 2, top + label_height + thickness + 1), rgb2bgr_(self.colors[c]), -1)

            # 绘制文本   anchor_point: bottom-left
            cv2.putText(image, label_text, (text_origin[0] + 1, text_origin[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0), thickness, lineType=cv2.LINE_AA)

        if show_:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            show_cvimage(image, 0)
        return image, top_boxes, top_conf, top_label

    def nms_cross_class(self, top_labels, top_confs, top_boxes, iou_threshold=0.85):
        nms_indices = []
        boxes_confs_labels = np.concatenate([top_boxes, top_confs[:, None], top_labels[:, None]], axis=1)
        sorted_indices = np.argsort(-boxes_confs_labels[:, 4])
        sorted_boxes_confs_labels = boxes_confs_labels[sorted_indices]
        while len(sorted_boxes_confs_labels) > 0:
            current_box = sorted_boxes_confs_labels[0]
            nms_indices.append(sorted_indices[0])
            if len(sorted_boxes_confs_labels) == 1:
                break
            ious = self.compute_iou(current_box[np.newaxis, :4], sorted_boxes_confs_labels[1:, :4])
            remaining_indices = np.where(ious < iou_threshold)[0]
            sorted_boxes_confs_labels = sorted_boxes_confs_labels[remaining_indices + 1]
            sorted_indices = sorted_indices[remaining_indices + 1]
        top_labels = top_labels[nms_indices]
        top_confs = top_confs[nms_indices]
        top_boxes = top_boxes[nms_indices]
        return top_labels, top_confs, top_boxes
    
    def compute_iou(self, box, boxes):
        x1 = np.maximum(box[:, 0], boxes[:, 0])
        y1 = np.maximum(box[:, 1], boxes[:, 1])
        x2 = np.minimum(box[:, 2], boxes[:, 2])
        y2 = np.minimum(box[:, 3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        iou = intersection / union
        return iou
        
    def detect_slide_pilimage(self, pilimage, show_=False, crop_size_=(2160,1200), rows_cols_=(2,2)):
        # pilimage  = resize_myimage(pilimage, (1280, 960), self.letterbox_image)
        imageBlocks = SlideImage()
        imageBlocksDict = imageBlocks.pillow_slide(pilimage, crop_size=crop_size_, cols_num=rows_cols_[1], rows_num=rows_cols_[0])
        image_shape = [imageBlocks.crop_size_h, imageBlocks.crop_size_w]
        results_list = []
        for rc_idx, iblock in imageBlocksDict.items():
            image_data  = resize_myimage(iblock, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
            # image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(iblock, dtype='float32')), (2, 0, 1)), 0)
            row_idx, col_idx = rc_idx
            with torch.no_grad():
                images = torch.from_numpy(image_data)
                if self.cuda:
                    images = images.cuda()

                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)

                if results[0] is None: 
                    continue

                # results ndarray 
                top_label   = np.array(results[0][:, 5], dtype = 'int32')
                top_conf    = results[0][:, 4]
                top_boxes   = results[0][:, :4]       # top, left, bottom, right    y1,x1 y2,x2
                
                top_boxes[:,0] += imageBlocks.stride_h * row_idx   # y1
                top_boxes[:,1] += imageBlocks.stride_w * col_idx  # x1
                top_boxes[:,2] += imageBlocks.stride_h * row_idx  # y2
                top_boxes[:,3] += imageBlocks.stride_w * col_idx  # x2
                for i in range(len(top_label)):
                    results_list.append([top_label[i], top_conf[i], top_boxes[i]])

        merged_results = merge_overlapped_ltbrBbox(results_list)

        if merged_results is not None:
            # # draw merged_result on image
            # pilimage_new = pilimage.copy()
            font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * pilimage.size[1] + 0.5).astype('int32'))
            thickness   = int(max((pilimage.size[0] + pilimage.size[1]) // np.mean(self.input_shape), 1))
            for label, score, box in merged_results:
                class_id = int(label)
                predicted_class = self.class_names[class_id]
                top, left, bottom, right = box

                # Ensure box coordinates are within image boundaries
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(pilimage.size[1], np.floor(bottom).astype('int32'))
                right = min(pilimage.size[0], np.floor(right).astype('int32'))

                # Create the label string
                # label_text = f'{predicted_class} {score:.2f}'
                label_text = f'{predicted_class}'
                draw = ImageDraw.Draw(pilimage)
                label_size = (
                draw.textbbox((0, 0), label_text, font=font)[2] - draw.textbbox((0, 0), label_text, font=font)[0], \
                draw.textbbox((0, 0), label_text, font=font)[3] - draw.textbbox((0, 0), label_text, font=font)[1]) \
                    if check_version(PIL.__version__, '9.2.0') else draw.textsize(label_text, font)

                label = label_text.encode('utf-8')
                # print(label_text, top, left, bottom, right)

                # Determine the text origin position
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # Draw the bounding box with the specified thickness
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[class_id])

                # Draw the label background and text
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[class_id])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                
                del draw
            if show_:
                pilimage.show()
            return pilimage
        else:
            if show_:
                pilimage.show()
            return pilimage

    def detect_slide_cvimage(self, cvimage, show_=False, slide_test_= False, crop_size_=[1920, 1080], rows_cols_=(2,2)):
        # cvimage = resize_myimage(cvimage, (1280, 960), self.letterbox_image)
        imageBlocks = SlideImage()
        imageBlocksDict = imageBlocks.opencv_slide(cvimage, drawBoxForTest=slide_test_, crop_size=crop_size_, cols_num=rows_cols_[1], rows_num=rows_cols_[0])
        image_shape = [imageBlocks.crop_size_h, imageBlocks.crop_size_w]
        results_list = []
        for rc_idx, iblock in imageBlocksDict.items():
            image_data = resize_myimage(iblock, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
            image_data = np.expand_dims(np.transpose(preprocess_input(image_data.astype('float32')), (2, 0, 1)), 0)
            row_idx, col_idx = rc_idx
            with torch.no_grad():
                images = torch.from_numpy(image_data)
                if self.cuda:
                    images = images.cuda()

                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

                if results[0] is None:
                    continue

                # results ndarray
                top_label = np.array(results[0][:, 5], dtype='int32')
                top_conf = results[0][:, 4]
                top_boxes = results[0][:, :4]  # top, left, bottom, right    y1,x1 y2,x2

                top_boxes[:, 0] += imageBlocks.stride_h * row_idx  # y1
                top_boxes[:, 1] += imageBlocks.stride_w * col_idx  # x1
                top_boxes[:, 2] += imageBlocks.stride_h * row_idx  # y2
                top_boxes[:, 3] += imageBlocks.stride_w * col_idx  # x2
                for i in range(len(top_label)):
                    results_list.append([top_label[i], top_conf[i], top_boxes[i]])

        merged_results = merge_overlapped_ltbrBbox(results_list)

        if merged_results is not None:

            # # draw merged_result on image
            # font_scale = np.floor(3e-2 * cvimage.shape[0] + 0.5).astype('int32')
            # thickness = int(max((cvimage.shape[0] + cvimage.shape[1]) // np.mean(self.input_shape), 1))
            font_scale = 2
            thickness = 4
            print(f"{font_scale=} {thickness=}")
            for label, score, box in merged_results:
                class_id = int(label)
                predicted_class = self.class_names[class_id]
                top, left, bottom, right = box

                # Ensure box coordinates are within image boundaries
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(cvimage.shape[0], np.floor(bottom).astype('int32'))
                right = min(cvimage.shape[1], np.floor(right).astype('int32'))

                # Create the label string
                # label_text = f'{predicted_class} {score:.2f}'
                label_text = f'{predicted_class}'
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                # bbox
                cv2.rectangle(cvimage, (left, top), (right, bottom), rgb2bgr_(self.colors[class_id]), thickness=thickness)

                # label = label_text.encode('utf-8')
                # # print(label_text, top, left, bottom, right)

                if top - label_size[1] >= 0:
                    # bottom-left
                    text_origin = np.array([left, top])
                else:
                    text_origin = np.array([left, top + 3])

                cv2.rectangle(cvimage, (text_origin[0] - 1, text_origin[1] - label_size[1] - 1),
                              (text_origin[0] + label_size[0] + 1, text_origin[1] + 1), rgb2bgr_(self.colors[class_id]), -1)
                cv2.putText(cvimage, label_text, (text_origin[0], text_origin[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 0), thickness // 2, lineType=cv2.LINE_AA)
            if show_:
                cvimage = cv2.cvtColor(cvimage, cv2.COLOR_RGB2BGR)
                show_cvimage(cvimage, 0)
            return cvimage
        else:
            if show_:
                cvimage = cv2.cvtColor(cvimage, cv2.COLOR_RGB2BGR)
                show_cvimage(cvimage, 0)
            return cvimage
   
    def detect_video_pyav(self, input_path, output_path, fps=None, show_=False):
        
        # 打开输入视频文件
        container = av.open(input_path)
        stream = container.streams.video[0]

        # 获取输入视频的属性
        width = stream.width
        height = stream.height
        original_fps = float(stream.average_rate)

        if fps is None:
            fps = original_fps

        # 创建输出视频容器
        output_container = av.open(output_path, mode='w')
        output_stream = output_container.add_stream('h264', rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuv420p'

        frame_interval = 1.0 / fps  # 帧间隔
        last_time = time.time()
        frame_count = 0

        for packet in container.demux(stream):
            for frame in packet.decode():
                current_time = time.time()
                elapsed_time = current_time - last_time
                if elapsed_time >= frame_interval:
                    last_time = current_time

                    # 将帧转换为 numpy 数组（BGR 格式）
                    img = frame.to_ndarray(format='bgr24')    # img = frame.to_ndarray(format='rgb24')
                    # rgb
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # 调用目标检测模型处理帧
                    result_frame = self.detect_cvimage(img)[0]

                    if show_:
                        result_frame_ = result_frame.copy()
                        result_frame_ = cv2.cvtColor(result_frame_, cv2.COLOR_RGB2BGR)
                        # 显示处理后的帧
                        continue_processing = show_cvimage(result_frame_, mode=1, title='Processing', w=width, h=height)
                        if not continue_processing:
                            # print("检测到按键 'q'，退出程序")
                            # 释放资源并退出
                            container.close()
                            output_container.close()
                            return

                    # 将处理后的帧转换回 VideoFrame（yuv420p 格式）
                    # result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    new_frame = av.VideoFrame.from_ndarray(result_frame, format='rgb24')
                    new_frame = new_frame.reformat(width, height, format='yuv420p')

                    # 编码并写入输出容器
                    for packet in output_stream.encode(new_frame):
                        output_container.mux(packet)

                    frame_count += 1
                    print(f"已处理第 {frame_count} 帧")

        # 刷新编码器缓冲区
        for packet in output_stream.encode():
            output_container.mux(packet)

        # 关闭容器和窗口
        container.close()
        output_container.close()
        cv2.destroyAllWindows()
        print(f"处理完成并保存到：{os.path.basename(output_path)}")
    
    def process_video_opencv(self, input_path, output_path, fps=None, show_=True):
        pass
    
    
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_myimage(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            if results[0] is None: 
                return
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
            top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.8)
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
        f.close()
        return

    def get_fps(self):
        "to do ..."
        pass

if __name__ == '__main__':
    from PIL import Image
    import cv2 
    slide_ = SlideImage()
    
    im_path = "img/a.jpg"
    cropsz = (1920,1080)
    r_, c_ = 2, 2
    # pillow
    pilimage = Image.open(im_path).convert('RGB')
    slide_.pillow_slide(pilimage, cropsz, r_, c_, drawBoxForTest=True)
    
    # # # opencv
    # cvimage = cv2.imread(im_path)
    # slide_.opencv_slide(cvimage, cropsz, r_, c_, drawBoxForTest=True)