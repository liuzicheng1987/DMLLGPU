export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
sudo ldconfig /usr/local/cuda-7.5/lib64
#nvcc -std=c++11 test.cu -arch=sm_20 -o test
#sudo optirun ./test
