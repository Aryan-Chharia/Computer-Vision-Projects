import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import easyocr
import cv2

# Set up paths and check for GPU availability
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative path

# Import YOLO-related modules
from models.common import DetectMultiBackend # type: ignore
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams # type: ignore
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, # type: ignore
                           colorstr, increment_path, non_max_suppression, print_args, scale_boxes,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box # type: ignore
from utils.torch_utils import select_device, smart_inference_mode # type: ignore

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

def perform_ocr_on_image(img, coordinates):
    """
    Perform OCR on a cropped region of the image based on the given coordinates.
    """
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)

    # Extract text based on specific conditions
    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]

    return str(text)

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # Model path or Triton URL
        source=ROOT / 'data/images',  # File/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # Dataset.yaml path
        imgsz=(640, 640),  # Inference size (height, width)
        conf_thres=0.25,  # Confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # Maximum detections per image
        device='',  # CUDA device
        view_img=False,  # Show results
        save_txt=False,  # Save results to *.txt
        save_conf=False,  # Save confidences in --save-txt labels
        save_crop=False,  # Save cropped prediction boxes
        nosave=False,  # Do not save images/videos
        classes=None,  # Filter by class
        agnostic_nms=False,  # Class-agnostic NMS
        augment=False,  # Augmented inference
        visualize=False,  # Visualize features
        update=False,  # Update all models
        project=ROOT / 'runs/detect',  # Save results to project/name
        name='exp',  # Save results to project/name
        exist_ok=False,  # Existing project/name ok
        line_thickness=3,  # Bounding box thickness (pixels)
        hide_labels=False,  # Hide labels
        hide_conf=False,  # Hide confidences
        half=False,  # Use FP16 half-precision inference
        dnn=False,  # Use OpenCV DNN for ONNX inference
        vid_stride=1,  # Video frame-rate stride
):
    # Check and prepare the source
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # Save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # Download

    # Set up directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # Increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # Make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # Check image size

    # Dataloader setup
    bs = 1  # Batch size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # Warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # Expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # Non-Maximum Suppression
        with dt[2]:
            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # Per image
            seen += 1
            if webcam:  # Batch size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # To Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # Print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
            imc = im0.copy() if save_crop else im0  # For save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # Detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # Label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # Integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        # Perform OCR and update the label
                        text_ocr = perform_ocr_on_image(im0, xyxy)
                        if text_ocr:
                            label = f'{label}, OCR: {text_ocr}' if label else f'OCR: {text_ocr}'

                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # Save cropped prediction boxes
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_path, BGR=True)

            # Print and save results
            LOGGER.info(f'{s}{colorstr("Done. ")}, {os.times:.3f}s')
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # Video
                    if vid_path[i] != save_path:  # New video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # Release previous video writer
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w, h = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Release video writers
    for i in range(bs):
        if isinstance(vid_writer[i], cv2.VideoWriter):
            vid_writer[i].release()

    # Print total processing time
    t = Profile()
    t.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', type=int, nargs='+', help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()

    print_args(vars(opt))  # Print args
    run(**vars(opt))  # Start inference
