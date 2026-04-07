"""
This file contains functions for running CLIP inference.

E6692 Spring 2026

YOU DO NOT NEED TO MODIFY THESE FUNCTIONS TO COMPLETE THE LAB
"""

import torch
import cv2
import io
from PIL import Image
import IPython

from .clip_utils import clip_prediction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2
COORD = (50, 50)
RADIUS = 5


def show_array(a, fmt='jpeg'):
    """
    Display array in Jupyter cell output using ipython widget.

    params:
        a (np.array): the input array
        fmt='jpeg' (string): the extension type for saving. Performance varies
                             when saving with different extension types.
    """
    f = io.BytesIO() # get byte stream
    Image.fromarray(a).save(f, fmt) # save array to byte stream
    display(IPython.display.Image(data=f.getvalue())) # display saved array

def clip_webcam_inference(model, preprocess, text_features, class_names, device = DEVICE):
    """
    Perform webcam inference using CLIP with custom defined classes.
    
    params:
        model: CLIP model
        preprocess: CLIP data preprocessor
        text_features: pre-computed text features.
        class_names: list of custom defined classes (list of strings)
        device: computing device. cpu / cuda
    returns:
        None
    """
    
    cam = cv2.VideoCapture(0) # define camera stream
    
    try:
        print("Video feed started.")

        while True:
            _, frame = cam.read() # read frame from video stream
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert raw frame from BGR to RGB
            
            im_pil = Image.fromarray(frame)
            
            class_probabilities, most_probable_class = clip_prediction(model, preprocess, im_pil, text_features)
            
            class_probabilities = class_probabilities.detach().cpu()
            most_probable_class = most_probable_class.detach().cpu()
            
            display_text = f"{class_names[most_probable_class]}: {class_probabilities[0][most_probable_class].item()}"
            frame = cv2.putText(frame, display_text, COORD, FONT,
                       FONT_SCALE, COLOR, THICKNESS, cv2.LINE_AA)
            
            show_array(frame) # display the frame in JupyterLab
            print(f'Predicted class: "{class_names[most_probable_class]}" \n')
            print('Class Probabilities:')
            for cls_name, cls_prob in zip(class_names, class_probabilities[0]):
                print(f'{cls_name.capitalize()}: {round(cls_prob.item(), 4)}')
                
            IPython.display.clear_output(wait=True) # clear the previous frame

    except Exception as e: # if interrupted
        print("Video feed stopped.")
        cam.release() # release the camera feed
