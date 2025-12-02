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

**2. --Arguments**

**You can use any of the following arguments with source.**

1.`--masks` will return masks only.

2.`--background` will return objects with no background.

3. `--bg_image` will return the object with provided image as background 

Note: need to use `--background` and `--bg_image`

4.`--show` is used to preview video other video will only save

5.`--closest` will only show the largest segmented object
  
**3.Run Examples:**
```python 
python main.py --source umair2.jpeg

python main.py --source umair2.jpeg --masks --background  --bg_image bg.jpg

python main.py --source umair2.jpeg --masks --bg_image bg.jpg

python main.py --source test.mp4 --masks --show

python main.py --source test.mp4 --masks
