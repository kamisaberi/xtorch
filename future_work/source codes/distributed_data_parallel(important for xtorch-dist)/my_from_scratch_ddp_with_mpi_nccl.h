#pragma once

#include "distributed_data_parallel_base.h" // Your base class
#include <mpi.h>    // For MPI functions
#include <nccl.h>   // For NCCL functions
#include <cuda_runtime.h> // For CUDA device management and streams
#include <string>
#include <vector>
#include <stdexcept> // For runtime_error

// Helper for NCCL/CUDA error checking (can be in a common utils header)
#define DDP_NCCLCHECK(cmd) do { \
  ncclResult_t r = cmd; \
  if (r!= ncclSuccess) { \
    fprintf(stderr, "NCCL failure %s:%d '%s'\n",__FILE__,__LINE__,ncclGetErrorString(r)); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
  } \
} while(0)

#define DDP_CUDACHECK(cmd) do { \
  cudaError_t e = cmd; \
  if( e != cudaSuccess ) { \
    fprintf(stderr,"CUDA failure %s:%d '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
  } \
} while(0)


namespace xt {
namespace nn {
namespace parallel {

    class MyFromScratchDDPWithMPIAndNCCL : public DistributedDataParallelBase {
    public:
        /**
         * @brief Constructor
         * @param module_to_wrap The xt::Module to parallelize.
         * @param assigned_device The torch::Device this rank is responsible for (must be a CUDA device if using NCCL).
         * @param mpi_comm The MPI communicator (e.g., MPI_COMM_WORLD).
         */
        MyFromScratchDDPWithMPIAndNCCL(std::shared_ptr<xt::Module> module_to_wrap,
                                       torch::Device assigned_device,
                                       MPI_Comm mpi_comm = MPI_COMM_WORLD); // Default to MPI_COMM_WORLD

        ~MyFromScratchDDPWithMPIAndNCCL() override;

        // --- Implement Pure Virtual Methods from Base ---
        std::any forward(std::initializer_list<std::any> inputs) override;
        void synchronize_gradients() override;

        // --- Overridden xt::Module methods for device handling ---
        void to(torch::Device device, bool non_blocking = false) override;
        // to(ScalarType) and to(Device, ScalarType) can use base or be overridden if specific logic needed


    private:
        void initialize_distributed_env(MPI_Comm mpi_comm);
        void broadcast_initial_parameters();
        void setup_nccl_communicator();

        std::shared_ptr<xt::Module> module_ref_for_params_; // Holds the module whose params are used by optimizer
                                                            // and is synced. This is underlying_module_ptr_ from base.

        // MPI specific
        MPI_Comm mpi_communicator_;

        // NCCL specific
        ncclComm_t nccl_communicator_;
        cudaStream_t nccl_stream_;

        bool initial_parameters_broadcasted_ = false;
        bool is_single_process_mode_ = false; // True if world_size is 1
    };

} // namespace parallel
} // namespace nn
} // namespace xt