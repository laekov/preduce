#!/bin/bash

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -std=c++11 -shared preduce.cc ../src/preduce.o -o preduce.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I../src -L../src -O2

