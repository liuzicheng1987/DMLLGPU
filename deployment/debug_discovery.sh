#Set path in which you want your Python modules copied. num_samplesf you use Python 2.7, then /usr/lib/python2.7/ is a good choice.
export PATH_FOR_PYTHON_MODULES=$(python find_path_for_python_modules.py)
export NUMPY_PATH=$(python find_numpy_path.py)
export COMPUTE_CAPABILITY=compute_50

#https://groups.google.com/forum/#!topic/mpi4py/Homqi5iJmT4
#However, it would be better to update your setup.py to make distutils build the SWIG module for you, instead of running SWIG manually. Or use a makefile, see for example demo/wrap-swig in the mpi4py sources.

#Compile C++/CUDA sourcecode and create discoveryCpp.py
swig -c++ -python DiscoveryCpp.i
g++ -std=c++11 -c DiscoveryCpp_wrap.cxx -g -I/usr/include/python2.7 -I$NUMPY_PATH -fPIC 
g++ -shared DiscoveryCpp_wrap.o -o _DiscoveryCpp.so

#If there is trouble loading cuBLAS:
#sudo ldconfig /usr/local/cuda-7.5/lib64
#https://www.kaggle.com/c/diabetic-retinopathy-detection/forums/t/15496/help-using-lasagne-in-ec2-theano-unable-to-detect-gpu


#Compile discovery.py and discoveryCpp.py
#If file already exists, delete
#if [ -f  discoveryCpp.pyc ];
#then
#	rm discoveryCpp.pyc
#   echo "Deleted discoveryCpp.pyc"
#fi

#if [ -f  discovery.pyc ];
#then
#	rm discovery.pyc
#   echo "Deleted discovery.pyc"
#fi

#python -m compileall discoveryCpp.py
#python -m compileall discovery.py

#Copy to Python module folder
#echo "In order to copy the modules into the appropriate python path, you might be asked to enter your sudo password..."
#sudo cp discoveryCpp_wrap.o $PATHFORPYTHONMODULES
#sudo cp _discoveryCpp.so $PATHFORPYTHONMODULES
#sudo cp discovery.py $PATHFORPYTHONMODULES
#sudo cp discoveryCpp.py $PATHFORPYTHONMODULES
#sudo cp discovery.pyc $PATHFORPYTHONMODULES
#sudo cp discoveryCpp.pyc $PATHFORPYTHONMODULES
