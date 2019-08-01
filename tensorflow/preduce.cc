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
        const Tensor& group_tensor = context->input(0);

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
                    input_tensor.shape(), &output_tensor));

        preduce::preduce(
                (const float*)(const void*)input_tensor.tensor_data().data(), 
                (const int*)(const void*)group_tensor.tensor_data().data(),
                input_tensor.tensor_data().size(), 
                (float*)(void*)output_tensor->tensor_data().data()
                );
    }
};

REGISTER_KERNEL_BUILDER(Name("PReduce").Device(DEVICE_GPU), PReduce);


REGISTER_OP("PReduceSync")
    .Output("result: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->Scalar());
            return Status::OK();
    });


class PReduceSync: public OpKernel {
public:
    explicit PReduceSync(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        preduce::sync();
    }
};

REGISTER_KERNEL_BUILDER(Name("PReduceSync").Device(DEVICE_CPU), PReduceSync);
