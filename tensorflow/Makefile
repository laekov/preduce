TF_HOME ?= ${HOME}/.local/lib/python3.6/site-packages/tensorflow
INCLUDES ?= -I$(TF_HOME)/include -I../src

CC ?= icc
CCFLAGS ?= -std=c++11

preduce.so : preduce.cc
	bash build.sh

%.o : %.cc
	$(CC) $(CCFLAGS) $< $(INCLUDES) -c -o $@
