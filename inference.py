import time
import os
import av
import cv2
import numpy as np
from PIL import Image
from yolo_for_infer import YOLO_INFER, show_cvimage
# from utils.utils import resize_myimage, load_cfg
# cfg = load_cfg()

if __name__ == "__main__":

    yolo = YOLO_INFER()

    mode = "CLI_input"

    crop            = False
    count           = False
    
    # video_infer_pyav
    video_infer_pyav_input = r"D:\ddesktop\monitoring\datadata\belthelmet\1107_test\for_infer\output.mp4"
    video_infer_pyav_output = r"D:\ddesktop\monitoring\datadata\belthelmet\1107_test\for_infer\225_2_infer.mp4"
    pyav_output_save = True 
    pyav_show_ = True
    pyav_fps = None
    
    # video_infer_opencv
    video_infer_opencv_input = r""
    video_infer_opencv_output = r""
    opencv_output_save = False
    opencv_fps = None
    opencv_show_ = True
    
    # video
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    
    # fps
    test_interval   = 100
    fps_image_path  = "img/street.jpg"

    # dir_predict
    dir_origin_path = r"D:\ddesktop\monitoring\datadata\belthelmet\1107_test\img_test"
    dir_save_path   = r"D:\ddesktop\monitoring\datadata\belthelmet\1107_test\img_test"

    # heatmap
    heatmap_save_path = "model_data/heatmap_vision.png"

    # export_onnx
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "CLI_input":
        while True:
            img = input('Input image filename:')
            try:
                # # # pillow
                # image = Image.open(img)
                
                # opencv
                image = cv2.imread(img)  # bgr image
                
            except:
                print('Open Error! Try again!')
                continue
            else:
                # # # pillow
                # # rr_image = yolo.detect_image(image, crop = crop, count=count)
                # rr_image = yolo.detect_slide_pilimage(image, show_=True)

                
                # # # # opencv
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # --> rgb
                # # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # --> bgr 
                # # show_cvimage(image, 0)
                # # r_image = yolo.detect_cvimage(image, True)[0]
                r_image = yolo.detect_slide_cvimage(image, True)

                
    elif mode == "predict":
        # # pillow
        # image = Image.open(r"F:\bianse\testdata\newnew\hxq_91z.png")
        # assert image is not None, "图片路径错误或文件不存在！"
        # r_image = yolo.detect_image(image)
        # r_image.show()
        
        # opencv 
        image = cv2.imread(f"")  # rgb
        assert image is not None, "图片路径错误或文件不存在！"
        r_image = yolo.detect_cvimage(image)[0]
        show_cvimage(r_image, 0)

         
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path).convert('RGB')
                r_image     = yolo.detect_image(image)
                # r_image     = yolo.detect_slide_pilimage(image)
                # r_image = yolo.detec_slideImage(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "video_infer_opencv":
        if os.path.basename(video_infer_opencv_input).lower().endswith('.mp4'):
            capture = cv2.VideoCapture(video_infer_opencv_input)
            if not capture.isOpened():
                print(f"无法打开视频文件：{video_infer_opencv_input}")

            # 获取视频的帧率和尺寸
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 设置输出视频的路径和编码格式
            dir_save_path = os.path.dirname(video_infer_opencv_output)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'，根据需求选择
            out = cv2.VideoWriter(video_infer_opencv_output, fourcc, fps, (width, height))


            count = 0
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                # 转换为 RGB 格式（如果需要）
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 进行目标检测
                r_image, _, _, _ = yolo.detect_cvimage(frame)

                # 将处理后的帧写入输出视频
                out.write(r_image)
                count += 1
                print(f"write{count} frame")

            # 释放资源
            capture.release()
            out.release()
            print(f"处理完成并保存到：{os.path.basename(video_infer_opencv_output)}")

    elif mode == "video_infer_pyav":
        if os.path.basename(video_infer_pyav_input).lower().endswith('.mp4'):
            
            # 检查输入视频文件是否存在
            assert os.path.exists(video_infer_pyav_input), f"视频文件不存在：{video_infer_pyav_input}"

            input_container = av.open(video_infer_pyav_input)
            input_stream = input_container.streams.video[0]
            original_fps = float(input_stream.average_rate)
            width = input_stream.width
            height = input_stream.height
            pix_fmt = input_stream.format.name 
            
            if pyav_fps is None:
                pyav_fps = original_fps 

            dir_save_path = os.path.dirname(video_infer_pyav_output)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            output_container = av.open(video_infer_pyav_output, mode='w')
            output_stream = output_container.add_stream('h264', rate=pyav_fps)
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = 'yuv420p'

            count = 0
            for frame in input_container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                r_image = yolo.detect_cvimage(img)[0]
                
                if pyav_show_:
                    continue_processing = show_cvimage(r_image, 1)
                    if not continue_processing:
                        input_container.close()
                        output_container.close()
                        exit()                   
                
                r_image_yuv = cv2.cvtColor(r_image, cv2.COLOR_RGB2YUV_I420)
                new_frame = av.VideoFrame.from_ndarray(r_image_yuv, format='yuv420p')
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                packet = output_stream.encode(new_frame)
                if packet:
                    output_container.mux(packet)

                count += 1
                print(f"已处理第 {count} 帧")

            # 刷新编码器缓冲区
            packet = output_stream.encode(None)
            if packet:
                output_container.mux(packet)

            input_container.close()
            output_container.close()
            cv2.destroyAllWindows()
            print(f"处理完成并保存到：{os.path.basename(video_infer_pyav_output)}")

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)
           
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
