#include "preduce.hh"

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <vector>
#include <unordered_map>


namespace preduce {

struct VectorHasher {
    int operator()(const std::vector<int> &v) const {
        unsigned hash = 0;
        for (auto& i : v) {
            hash = hash * 17 + i;
        }
        return (int)hash;
    }
};


std::unordered_map<std::vector<int>, ncclComm_t, VectorHasher> comms;
std::unordered_map<std::vector<int>, MPI_Comm, VectorHasher> mpicomms;
int comm_world_size, comm_world_rank;
cudaStream_t stream;
bool initialized(false);

void init(bool cuda=true) {
    MPI_Init(0, 0);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);

    if (cuda) {
        CUDACHECK(cudaStreamCreate(&stream));
    }
    initialized = true;
}

void preduce_cpu(const float* inbuf, const int* group, int n, float* outbuf) {
    if (!initialized) {
        init(false);
    }
 
    int comm_rank = 0, comm_size = 0;
    std::vector<int> members;
    for (int i = 0; i < comm_world_size; ++i) {
        if (group[i]) {
            members.push_back(i);
        }
    }
    auto it(mpicomms.find(members));

    MPI_Comm allred_comm;

    if (it == mpicomms.end()) {
        int comm_size(members.size()) ;
        MPI_Group world_group, allred_group;
        ncclUniqueId id;

        MPICHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
        MPICHECK(MPI_Group_incl(world_group, comm_size, members.data(), 
                    &allred_group));
        MPICHECK(MPI_Comm_create_group(MPI_COMM_WORLD, allred_group, 0, 
                    &allred_comm));
        mpicomms[members] = allred_comm;
    } else {
        allred_comm = it->second;
    }
    fprintf(stderr, "%d do die %f\n", comm_world_rank, *inbuf);
    MPI_Comm_rank(allred_comm, &comm_rank);
    MPICHECK(MPI_Allreduce(inbuf, outbuf, n, MPI_FLOAT, MPI_SUM, allred_comm));
    fprintf(stderr, "allreduce done\n");
}

void preduce_gpu(const float* inbuf, const int* group, int n, float* outbuf) {
    if (!initialized) {
        init();
    }

    int comm_rank = 0, comm_size = 0;

    int* group_cpu = new int[comm_world_size];
    cudaMemcpy(group_cpu, group, sizeof(int) * comm_world_size,
            cudaMemcpyDeviceToHost);
    std::vector<int> members;
    for (int i = 0; i < comm_world_size; ++i) {
        if (group_cpu[i]) {
            members.push_back(i);
        }
    }

    auto it(comms.find(members));
    ncclComm_t comm;
    if (it == comms.end()) {
        int comm_size(members.size()) ;
        MPI_Group world_group, allred_group;
        ncclUniqueId id;

        MPICHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
        MPICHECK(MPI_Group_incl(world_group, comm_size, members.data(), 
                    &allred_group));
        MPI_Comm allred_comm;
        MPICHECK(MPI_Comm_create_group(MPI_COMM_WORLD, allred_group, 0, 
                    &allred_comm));
        MPICHECK(MPI_Comm_rank(allred_comm, &comm_rank));
        if (comm_rank == 0) { 
            ncclGetUniqueId(&id); 
        }
        MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, allred_comm));
        NCCLCHECK(ncclCommInitRank(&comm, comm_size, id, comm_rank));
        comms[members] = comm;
    } else {
        comm = it->second;
    }
    NCCLCHECK(ncclAllReduce((void*)inbuf, (void*)outbuf, n, ncclFloat, ncclSum,
               comm, stream));

    delete [] group_cpu;
}

void sync() {
    CUDACHECK(cudaStreamSynchronize(stream));
}

};  // namespace preduce


extern "C" {
void preduceSync() {
    preduce::sync();
}

void preduceCompute(const float* a, const int* b, int c, float* d) {
    preduce::preduce_cpu(a, b, c, d);
    preduce::sync();
}
};
