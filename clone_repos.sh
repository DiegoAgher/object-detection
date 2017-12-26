
# Install keras-resnet
pip install git+https://github.com/broadinstitute/keras-resnet

# Install keras-retinanet
git clone https://github.com/fizyr/keras-retinanet.git keras_retinanet
cd keras_retinanet
touch __init__.py
# python setup.py install
cd ..


git clone https://github.com/DiegoAgher/object-detection.git
cd object-detection
pip install -r requirements.txt
cd training/models_weights/
wget https://www.dropbox.com/s/2snae87k61qagfa/resnet50_csv_1200_10_continuation.h5
cd ../..
