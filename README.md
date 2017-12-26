# object-detection
Object detection with deep learning models

# Docker build
docker build -t obj_rec --no-cache .
docker run -p 8888:8888 obj_rec

# Inferencing
After building docker open a browser to acces jupyter notebook. The notebook `Inferencing.ipynb` has the interface to do inference.
