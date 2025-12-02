from models.segment_model import YoloSegmentation
import numpy as np
import cv2

def run_segmentation(args):
    if args.source is None:
        raise SystemExit('Provide an Image, video or Webcam')

    model = YoloSegmentation()
    source = args.source

    PALETTE = np.array([
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [128, 0, 128],
        [0, 128, 255],
        [128, 128, 0],
        [0, 128, 128],
    ], dtype=np.uint8)

    def pick_colors(n):
        if n <= 0:
            return np.zeros((1,3), dtype=np.uint8)
        reps = (n + len(PALETTE) - 1) // len(PALETTE)
        cols = np.vstack([PALETTE] * reps)[:n]
        return cols

    def prepare_mask_bins_and_areas(masks, w, h):
        mask_bins = []
        areas = []
        for mask_t in masks:
            mask_np = mask_t.cpu().numpy()
            mask_rs = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_rs > 0.5).astype('uint8')
            mask_bins.append(mask_bin)
            areas.append(int(mask_bin.sum()))
        return mask_bins, areas

    def choose_indices(areas, keep_closest):
        if keep_closest:
            if len(areas) > 0:
                return [int(np.argmax(areas))]
            return []
        return list(range(len(areas)))

    def render_masks_and_labels(mask_bins, keep_idx, model_names, r, w, h):
        colors = pick_colors(len(keep_idx))
        background_masks = np.zeros((h, w, 3), dtype=np.uint8)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        labels = []

        for ci, i in enumerate(keep_idx):
            mask_bin = mask_bins[i]
            colored_mask = mask_bin[..., None] * colors[ci]
            background_masks = np.maximum(background_masks, colored_mask)
            combined_mask = np.maximum(combined_mask, mask_bin)

            class_id = int(r.boxes.cls[i])
            label = model_names[class_id] if class_id in model_names else f"obj_{i}"

            ys, xs = np.where(mask_bin == 1)
            if len(xs) > 0:
                x_text = int(xs.min())
                y_text = int(max(16, ys.min() - 4))
                labels.append((label, x_text, y_text))
                cv2.putText(background_masks, label, (x_text, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return background_masks, combined_mask, labels

    def save_image_cutout(image, combined_mask, out_name, bg_replace_path=None):
        h, w = combined_mask.shape
        mask_bool = combined_mask.astype(bool)

        if bg_replace_path:
            bg = cv2.imread(bg_replace_path)
            if bg is None:
                raise SystemExit(f"Cannot read background image: {bg_replace_path}")
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
            composite = bg.copy()
            composite[mask_bool] = image[mask_bool]
            cv2.imwrite(out_name, composite)
        else:
            cutout = np.zeros_like(image, dtype=np.uint8)
            cutout[mask_bool] = image[mask_bool]
            alpha = (combined_mask * 255).astype(np.uint8)
            b, g, r_chan = cv2.split(image)
            bgra = cv2.merge([b, g, r_chan, alpha])
            cv2.imwrite(out_name, bgra)

    
    if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        image = cv2.imread(source)
        if image is None:
            raise SystemExit("Cannot read image")

        h, w = image.shape[:2]

        r = model.segmentation(image)
        masks = r.masks.data if r.masks is not None else []

        mask_bins, areas = prepare_mask_bins_and_areas(masks, w, h)
        keep_idx = choose_indices(areas, getattr(args, "closest", False))

        background_masks, combined_mask, labels = render_masks_and_labels(
            mask_bins, keep_idx, model.names, r, w, h
        )

        if getattr(args, "masks", False):
            cv2.imwrite('masks_output.jpg', background_masks)

        if getattr(args, "background", False):
            if getattr(args, "bg_image", None):
                save_image_cutout(image, combined_mask, 'cutout_with_bg.jpg', bg_replace_path=args.bg_image)
            else:
                save_image_cutout(image, combined_mask, 'cutout_transparent.png', bg_replace_path=None)

        return

    
    if source.lower() == 'webcam':
        capture = cv2.VideoCapture(0)
        is_webcam = True
    else:
        capture = cv2.VideoCapture(source)
        is_webcam = False

    if not capture.isOpened():
        raise SystemExit('Cannot Open webcam or video')

    out_masks = None
    out_cutout = None
    frame_count = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        r = model.segmentation(frame)
        masks = r.masks.data if r.masks is not None else []

        mask_bins, areas = prepare_mask_bins_and_areas(masks, w, h)
        keep_idx = choose_indices(areas, getattr(args, "closest", False))

        background_masks, combined_mask, labels = render_masks_and_labels(
            mask_bins, keep_idx, model.names, r, w, h
        )

        if out_masks is None and getattr(args, "masks", False):
            real_fps = capture.get(cv2.CAP_PROP_FPS)
            if real_fps is None or real_fps <= 0 or np.isnan(real_fps):
                real_fps = 20.0
            out_masks = cv2.VideoWriter('masks_video.mp4',
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        real_fps, (w, h))

        if out_cutout is None and getattr(args, "background", False):
            real_fps = capture.get(cv2.CAP_PROP_FPS)
            if real_fps is None or real_fps <= 0 or np.isnan(real_fps):
                real_fps = 20.0
            out_cutout = cv2.VideoWriter('cutout_video.mp4',
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         real_fps, (w, h))

        if out_masks is not None:
            out_masks.write(background_masks)

        mask_bool = combined_mask.astype(bool)
        cutout = np.zeros_like(frame, dtype=np.uint8)
        cutout[mask_bool] = frame[mask_bool]

       
        if getattr(args, "bg_image", None):
            bg_replace = cv2.imread(args.bg_image)
            if bg_replace is None:
                raise SystemExit(f"Cannot read background image: {args.bg_image}")
            bg_replace = cv2.resize(bg_replace, (w, h), interpolation=cv2.INTER_AREA)

            composite = bg_replace.copy()
            composite[mask_bool] = frame[mask_bool]

            if out_cutout is not None:
                out_cutout.write(composite)

            
            if is_webcam:
                cv2.imshow("Segmented Output", composite)
            elif getattr(args, "show", False):
                cv2.imshow("Segmented Output", composite)

        else:
            if out_cutout is not None:
                out_cutout.write(cutout)

            if is_webcam:
                cv2.imshow("Segmented Output", cutout)
            elif getattr(args, "show", False):
                cv2.imshow("Segmented Output", cutout)

        
        if is_webcam or getattr(args, "show", False):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested stop.")
                break

        frame_count += 1

    capture.release()
    if out_masks is not None:
        out_masks.release()
    if out_cutout is not None:
        out_cutout.release()

    if is_webcam or getattr(args, "show", False):
        cv2.destroyAllWindows()

    print(f"Written {frame_count} frames")
