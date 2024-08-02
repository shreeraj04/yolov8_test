from ultralytics import solutions
from streamlit_webrtc import webrtc_streamer
webrtc_streamer(key="example")
# Pass a model as an argument
solutions.inference(model="yolov8m.onnx")

### Make sure to run the file using command `streamlit run <file-name.py>
