CU ?= nvcc
CCFLAGS ?= -std=c++11
MPI_HOME ?= /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi
NCCL_HOME ?= /opt/apps/cuda9_0/nccl/2.2.13
PR_INCLUDES ?= -I$(MPI_HOME)/intel64/include -I$(NCCL_HOME)/include
PR_LIBS ?= -L$(MPI_HOME)/lib64 -L$(NCCL_HOME)/lib

libpreduce.a : preduce.o
	ar -cvq $@ $<

%.o : %.cu.cc
	$(CU) -Xcompiler -fPIC $(CCFLAGS) $< $(PR_INCLUDES) -c -o $@
