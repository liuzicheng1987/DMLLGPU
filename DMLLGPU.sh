#Set path in which you want your Python modules copied. num_samplesf you use Python 2.7, then /usr/lib/python2.7/ is a good choice.
export PATHFORPYTHONMODULES=$(python -c "import sys; print(sys.path[1])")

#https://groups.google.com/forum/#!topic/mpi4py/Homqi5iJmT4
#However, it would be better to update your setup.py to make distutils build the SWIG module for you, instead of running SWIG manually. Or use a makefile, see for example demo/wrap-swig in the mpi4py sources.

#Compile C++/CUDA sourcecode and create DMLLGPUCpp.py
swig -c++ -python DMLLGPUCpp.i
mv DMLLGPUCpp_wrap.cxx DMLLGPUCpp_wrap.cu 
nvcc -std=c++11 -ccbin g++ -m64 -gencode arch=compute_50,code=compute_50 -c DMLLGPUCpp_wrap.cu -lcublas -lcusparse -I/usr/include/python2.7 -Xcompiler -fPIC 
nvcc -gencode arch=compute_50,code=compute_50 -lcublas -lcusparse -shared DMLLGPUCpp_wrap.o -o _DMLLGPUCpp.so

#If there is trouble loading cuBLAS:
#sudo ldconfig /usr/local/cuda-7.5/lib64
#https://www.kaggle.com/c/diabetic-retinopathy-detection/forums/t/15496/help-using-lasagne-in-ec2-theano-unable-to-detect-gpu


#Compile DMLL.py and DMLLCpp.py
#If file already exists, delete
#if [ -f  DMLLCpp.pyc ];
#then
#	rm DMLLCpp.pyc
#   echo "Deleted DMLLCpp.pyc"
#fi

#if [ -f  DMLL.pyc ];
#then
#	rm DMLL.pyc
#   echo "Deleted DMLL.pyc"
#fi

#python -m compileall DMLLCpp.py
#python -m compileall DMLL.py

#Copy to Python module folder
#echo "In order to copy the modules into the appropriate python path, you might be asked to enter your sudo password..."
#sudo cp DMLLCpp_wrap.o $PATHFORPYTHONMODULES
#sudo cp _DMLLCpp.so $PATHFORPYTHONMODULES
#sudo cp DMLL.py $PATHFORPYTHONMODULES
#sudo cp DMLLCpp.py $PATHFORPYTHONMODULES
#sudo cp DMLL.pyc $PATHFORPYTHONMODULES
#sudo cp DMLLCpp.pyc $PATHFORPYTHONMODULES
