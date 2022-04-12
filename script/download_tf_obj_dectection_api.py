r"""Script to download the TensorFlow Object Detection API repository
and use protoc to translate .protos file to .py files for installation"""

%%bash
git clone --depth 1 https://github.com/tensorflow/models
sudo apt install -y protobuf-compiler
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python3 -m pip install .
