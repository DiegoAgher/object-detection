FROM ubuntu:16.04


ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LC_CTYPE=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN apt-get update
RUN apt-get install -y -q wget curl
RUN apt-get install -y -q build-essential cmake
RUN apt-get install -y -q git pkg-config vim
RUN apt-get install -y -q libavformat-dev libavcodec-dev libavfilter-dev libswscale-dev
RUN apt-get install -y -q libjpeg-dev libpng-dev libtiff-dev libjasper-dev zlib1g-dev libopenexr-dev libxine2-dev libeigen3-dev libtbb-dev


SHELL ["/bin/bash", "-c"]
RUN mkdir -p /usr/local/object_detection
WORKDIR /usr/local/object_detection

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O miniconda.sh && \
    chmod u+x miniconda.sh && ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# RUN echo "${PATH}"
# ENV PATH="${HOME}/miniconda/bin:${PATH}"
# RUN ls "${HOME}/miniconda/bin"
# RUN echo "${PATH}"
# RUN source "${HOME}/miniconda/bin/activate" obj_rec

RUN conda create --name obj_rec python=3.5
RUN conda install --name obj_rec -c menpo opencv tensorflow jupyter setuptools pip

COPY ./clone_repos.sh /usr/local/object_detection/
RUN chmod u+x /usr/local/object_detection/clone_repos.sh
RUN source /opt/conda/bin/activate obj_rec && ./clone_repos.sh
ENV PYTHONPATH=/usr/local/object_detection:$PYTHONPATH

COPY ./run.sh /usr/local/object_detection/
RUN chmod u+x ./run.sh

CMD ["/bin/bash", "run.sh"]


