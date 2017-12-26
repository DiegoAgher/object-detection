
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
cd ..
