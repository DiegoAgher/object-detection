# Download & build opencv

# WORKSPACE???
# opencv build flags
# opencv version

git clone git://github.com/Itseez/opencv.git
cd opencv
git checkout 3.3.1
cd ..

git clone git://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 3.3.1
cd ..

cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D INSTALL_C_EXAMPLES=ON \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -D BUILD_EXAMPLES=ON ..
make
make install
ldconfig

# BUILD_PYTHON_SUPPORT=ON -D WITH_XINE=ON -D WITH_TBB=ON ..


