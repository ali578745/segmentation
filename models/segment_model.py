from ultralytics import YOLO
class YoloSegmentation:
    def __init__(self, model_name = 'weights/yolo11x-seg.pt'):
        self.model_name = model_name
        self.model = YOLO(self.model_name)
        self.names = self.model.names


    def segmentation(self, frame):
        results = self.model(frame)
        return results[0]
