from models.segment_model import YoloSegmentation
import numpy as np
import cv2

def run_segmentation(args):
    if args.source is None:
        raise SystemExit('Provide an Image, video or Webcam')
    
    model = YoloSegmentation()
    source = args.source

    if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        image = cv2.imread(source)
        if image is None:
            raise SystemExit("Cannot read image")

        h, w = image.shape[:2]

        r = model.segmentation(image)

        masks = r.masks.data
        num_masks = len(masks)

        mask_bins = []
        areas = []
        for i, mask_t in enumerate(masks):
            mask_np = mask_t.cpu().numpy()
            mask_rs = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_rs > 0.5).astype('uint8')
            mask_bins.append(mask_bin)
            areas.append(int(mask_bin.sum()))

        # choose indices to keep
        if getattr(args, "closest", False):
            if len(areas) > 0:
                keep_idx = [int(np.argmax(areas))]
            else:
                keep_idx = []
        else:
            keep_idx = list(range(len(mask_bins)))

        background = np.zeros_like(image, dtype=np.uint8)
        colors = np.random.randint(0, 255, (max(1, len(keep_idx)), 3), dtype=np.uint8)
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        color_i = 0
        for i in keep_idx:
            mask_bin = mask_bins[i]
            colored_mask = mask_bin[..., None] * colors[color_i]
            color_i += 1
            background = np.maximum(background, colored_mask)
            combined_mask = np.maximum(combined_mask, mask_bin)

            class_id = int(r.boxes.cls[i])
            label = model.names[class_id]

            ys, xs = np.where(mask_bin == 1)
            if len(xs) > 0:
                x_text = xs.min()
                y_text = max(20, ys.min() - 5)
                cv2.putText(background, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 3)

        cv2.namedWindow('Masks', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Masks', 800, 600)
        cv2.imshow('Masks', background)
        cv2.waitKey(0)

        cv2.imwrite('masks_output.jpg', background)

        alpha = (combined_mask * 255).astype(np.uint8)
        b, g, r_chan = cv2.split(image)
        bgra = cv2.merge([b, g, r_chan, alpha])
        cv2.imwrite('cutout_transparent.png', bgra)

        cv2.destroyAllWindows()
        return    

    
    if source.lower() == 'webcam':
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(source)    
    if not capture.isOpened():
        raise SystemExit('Cannot Open webcam')
    
    width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = capture.get(cv2.CAP_PROP_FPS) or 30.0

    out_video = None
    out_video2 = None  
    frame_count = 0 
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        r = model.segmentation(frame)

        background = np.zeros_like(frame, dtype=np.uint8)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        masks = r.masks.data
        num_masks = len(masks)

    
        mask_bins = []
        areas = []
        for i, mask_t in enumerate(masks):
            mask_np = mask_t.cpu().numpy()
            mask_rs = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_rs > 0.5).astype('uint8')
            mask_bins.append(mask_bin)
            areas.append(int(mask_bin.sum()))

        
        if getattr(args, "closest", False):
            if len(areas) > 0:
                keep_idx = [int(np.argmax(areas))]
            else:
                keep_idx = []
        else:
            keep_idx = list(range(len(mask_bins)))

        
        colors = np.random.randint(0, 255, (max(1, len(keep_idx)), 3), dtype=np.uint8)
        labels_to_draw = []

        color_i = 0
        for i in keep_idx:
            mask_bin = mask_bins[i]
            colored_mask = mask_bin[..., None] * colors[color_i]
            color_i += 1
            background = np.maximum(background, colored_mask)
            combined_mask = np.maximum(combined_mask, mask_bin)
            class_id = int(r.boxes.cls[i])
            label = model.names[class_id]
            ys, xs = np.where(mask_bin == 1)
            if len(xs) > 0:
                x_text = int(xs.min())
                y_text = int(max(16, ys.min() - 4))
                labels_to_draw.append((label, x_text, y_text))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Webcam/Video stopped by user.")
            break

        if out_video is None:
            real_fps = capture.get(cv2.CAP_PROP_FPS)
            if real_fps is None or real_fps <= 0 or np.isnan(real_fps):
                real_fps = 20.0
            out_video = cv2.VideoWriter('masks_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), real_fps, (w, h))
            out_video2 = cv2.VideoWriter('masks_on_original.mp4', cv2.VideoWriter_fourcc(*'mp4v'), real_fps, (w, h))
            if not out_video.isOpened() or not out_video2.isOpened():
                capture.release()
                raise SystemExit("Failed to open VideoWriter for masks_video.mp4")
        
        blended = cv2.addWeighted(frame, 1.0, background, 0.6, 0)
        mask3 = combined_mask.astype(bool)
        output_frame = frame.copy()
        output_frame[mask3] = blended[mask3]
        
        for lbl, lx, ly in labels_to_draw:
            cv2.putText(output_frame, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Segmented Output", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        out_video.write(background)  
        out_video2.write(output_frame)

        frame_count += 1

    capture.release()
    if out_video is not None:
        out_video.release()
    if out_video2 is not None:  
        out_video2.release()
