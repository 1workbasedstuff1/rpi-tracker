# the repository uses uv and i need to fetch the 
# from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
#                             InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)
# dependency so this pulls in the code needed for that
/usr/bin/python3 -c "import hailo_platform; print(hailo_platform.__file__)"
deactivate 2>/dev/null
rm -rf .venv
uv venv --system-site-packages --python /usr/bin/python3
source .venv/bin/activate
uv pip install opencv-python numpy
python -c "import hailo_platform; print('ok')"

# python hailo_retinaface.py --hef retinaface_mobilenet_v1.hef
#python3 src/bin/hailo_rpi_detection.py --hef retinaface_mobilenet_v1_8l.hef
