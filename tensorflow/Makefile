TF_HOME ?= ${HOME}/.local/lib/python3.6/site-packages/tensorflow
INCLUDES ?= -I$(TF_HOME)/include -I../src

MPI_HOME ?= /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi
CUDA_HOME ?= /opt/apps/cuda/9.0
NCCL_HOME ?= /opt/apps/cuda9_0/nccl/2.2.13

TF_CFLAGS ?= $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') 
TF_LFLAGS ?= $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CXX ?= g++
CCFLAGS ?= -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0

preduce.so : tfpreduce.o ../src/preduce.o
	$(CXX) $(CCFLAGS) -shared tfpreduce.o ../src/preduce.o \
		-lmpi -lnccl -ltensorflow_framework \
		${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
		-I../src -L../src \
		-L$(MPI_HOME)/lib64 -L$(NCCL_HOME)/lib \
		-L/home1/03924/xuehaiq/.local/lib/python3.6/site-packages/tensorflow \
		-fPIC -o $@ 

tf%.o : %.cc
	$(CXX) -fPIC $(CCFLAGS) $< $(INCLUDES) -c -o $@

../src/preduce.o : ../src/preduce.cu.cc
	cd ../src && make preduce.o

clean : 
	rm -f *.o *.so
