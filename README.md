# object-detection
Object detection with deep learning models

# Docker build
In order to build the Docker container, first clone this repo.
After, run the following from the command line `docker build -t obj_rec --no-cache .` This could take some minutes depeding on the internet speed. A message of successful building should be displayed after the process is finished.

After a successful build, run the following to launch a Jupyter instance to explore the model: `docker run -p 8888:8888 obj_rec`

# Inferencing
Once Docker is running, open a browser to access Jupyter using the link provided on terminal incluiding the token. The notebook `Inferencing.ipynb` inside the `object_detection` directory contains an example of how to use the interface to do inference.
