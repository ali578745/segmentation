# YOLO Segmentation Project:
This project performs instance segmentation using a YOLO segmentation model.  

## Features:

It supports:
- Image segmentation
- Video segmentation
- Live webcam segmentation
- Optional mode to keep **only the closest person/object** using the `--closest` flag

##  Requirements:

- Python 3.10
- A working webcam (for `--webcam` mode)
- `pip` package manager
- All required Packages in requirements.txt

## Start Guide:
-Open Terminal

# 1. Create Virtual Environment:
```python 
python -m venv venv
venv\Scripts\activate
``` 
# 2. Install Dependencies:
```python
pip install -r requirements.txt
```
# 3. Project Structure:
```
project/
│
├── main.py
├── requirements.txt
│
├── models/
│   └── segment_model.py
│
├── utils/
│   └── image_processing.py
│
└── weights/
    └── yolo11x-seg.pt

```
# 4. Usage:
The script uses two main arguments:

**1. --source (required):**

Accepts:

Image path

Video path

"webcam"

**2. --closest (optional):**

If used, only the closest person/object is segmented.**

**3.Run on an Image:**
```python 
python main.py --source umair2.jpeg

```
**With closest-only segmentation:** 
```python 
python main.py --source umair2.jpeg --closest
```
**4.Run on a Video:**
```python
python main.py --source myvideo.mp4
```
**With closest-only segmentation:**
```python
python main.py --source myvideo.mp4 --closest
```

**5.Run on Webcam:**
```python
python main.py --source webcam
```
**With closest-only segmentation:**
```python
python main.py --source webcam --closest
```
# 5. Output Files:
```python
masks_video.mp4
```
```python
masks_on_original.mp4
```
```python
masks_output.jpg
```
```python
cutout_transparent.png
```
