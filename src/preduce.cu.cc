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
int comm_world_size;
cudaStream_t stream;
bool initialized(false);

void init() {
    MPI_Init(0, 0);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
    CUDACHECK(cudaStreamCreate(&stream));
    initialized = true;
}

void preduce(float* databuf, int* group, size_t n) {
    if (!initialized) {
        init();
    }

    int comm_rank = 0, comm_size = 0;
    std::vector<int> members;
    for (int i = 0; i < comm_world_size; ++i) {
        if (group[i]) {
            members.push_back(i);
        }
    }

    auto it(comms.find(members));
    ncclComm_t comm;
    if (it == comms.end()) {
        int comm_size(members.size()), comm_rank;
        MPI_Group world_group, allred_group;
        ncclUniqueId id;

        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group_incl(world_group, comm_size, members.data(), &allred_group);
        MPI_Comm allred_comm;
        MPI_Comm_create_group(MPI_COMM_WORLD, allred_group, 0, &allred_comm);
        MPI_Comm_rank(allred_comm, &comm_rank);
        if (comm_rank == 0) { 
            ncclGetUniqueId(&id); 
        }
        MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, allred_comm));
        NCCLCHECK(ncclCommInitRank(&comm, comm_size, id, comm_rank));
        comms[members] = comm;
    } else {
        comm = it->second;
    }
    NCCLCHECK(ncclAllReduce((void*)databuf, (void*)databuf, n, ncclFloat, ncclSum, comm, stream));
}

void sync() {
    CUDACHECK(cudaStreamSynchronize(stream));
}

};  // namespace preduce
