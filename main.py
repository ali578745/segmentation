from utils.image_processing import run_segmentation
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Image, Video or Webcam')
    parser.add_argument('--closest', action='store_true', help='Enable closest-person-only masking')
    parser.add_argument('--masks', action='store_true', help='Save mask-only output (black background + colored masks)')
    parser.add_argument('--background', action='store_true', help='Save background-removed output (objects only on black or replaced background)')
    parser.add_argument('--bg_image', type=str, default=None, help='Path to image used as replacement background (optional, use with --background)')
    parser.add_argument('--show', action='store_true', help='Show preview windows while processing (only applies to webcam)')
    args = parser.parse_args()

    if not args.masks and not args.background:
        parser.error("You must provide --masks and/or --background along with --source (cannot give only --source).")

    if args.bg_image and not args.background:
        parser.error("--bg_image is only valid when --background is also provided.")

    run_segmentation(args)

if __name__ == '__main__':
    main()
