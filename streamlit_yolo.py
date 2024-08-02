import streamlit as st
from ultralytics import solutions
from streamlit_webrtc import webrtc_streamer
st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
webrtc_streamer(key="example")
# Pass a model as an argument
solutions.inference(model="yolov8m.onnx")

### Make sure to run the file using command `streamlit run <file-name.py>
