import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (adjust the path to your model)
model = YOLO("yolov8m.onnx")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
    
    def transform(self, frame):
        # Convert the image to BGR format
        img = frame.to_ndarray(format="bgr24")
        
        # Perform YOLOv8 inference
        results = self.model(img)
        
        # Draw results on the image
        annotated_frame = results.plot()
        
        return annotated_frame

# Title of the app
st.title("YOLOv8 Object Detection with Streamlit")

# Create the WebRTC streamer
webrtc_streamer(key="yolov8", video_processor_factory=VideoProcessor)
