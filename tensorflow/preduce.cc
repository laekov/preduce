#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "preduce.hh"

using namespace tensorflow;

REGISTER_OP("PReduce")
    .Input("data: float32")
    .Input("group: int32")
    .Output("sum: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
    });


class PReduce: public OpKernel {
public:
    explicit PReduce(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        int data_size = static_cast<int>(input_tensor.NumElements());

        const Tensor& group_tensor = context->input(1);

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
                    input_tensor.shape(), &output_tensor));

        preduceCompute(
                input_tensor.flat<float>().data(), 
                group_tensor.flat<int>().data(), data_size,
                output_tensor->flat<float>().data()
                );
    }
};

REGISTER_KERNEL_BUILDER(Name("PReduce").Device(DEVICE_GPU), PReduce);
