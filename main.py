from utils.image_processing import run_segmentation
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Image, Video or Webcam')
    parser.add_argument('--closest',action='store_true',help='Enable closest-person-only masking')
    args = parser.parse_args()
    run_segmentation(args)
if __name__ == '__main__':
    main()    
