#ifndef PREDUCE_HH
#define PREDUCE_HH

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
        fprintf(stderr, "Failed: MPI error %s:%d '%d'\n",        \
                __FILE__,__LINE__, e);   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
        fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n",             \
                __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
        fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n",             \
                __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


extern "C" {
void preduceSync();
void preduceCompute(const float*, const int*, int, float*);
};

#endif  // PREDUCE_HH

