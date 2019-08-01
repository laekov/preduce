#ifndef PREDUCE_HH
#define PREDUCE_HH

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
        printf("Failed: MPI error %s:%d '%d'\n",        \
                __FILE__,__LINE__, e);   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",             \
                __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
        printf("Failed, NCCL error %s:%d '%s'\n",             \
                __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)

namespace preduce {
void preduce(const float*, const int*, int, float*);
void sync();
};  // namespace preduce


extern "C" {
void preduceSync();
void preduceCompute(const float*, const int*, int, float*);
};

#endif  // PREDUCE_HH

