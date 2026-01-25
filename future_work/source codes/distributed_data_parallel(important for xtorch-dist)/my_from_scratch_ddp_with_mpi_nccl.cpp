#include "my_from_scratch_ddp_with_mpi_nccl.h"
#include <iostream>
#include <torch/serialize.h> // For torch::save/load (though direct data_ptr used for broadcast)

namespace xt {
namespace nn {
namespace parallel {

    MyFromScratchDDPWithMPIAndNCCL::MyFromScratchDDPWithMPIAndNCCL(
        std::shared_ptr<xt::Module> module_to_wrap,
        torch::Device assigned_device,
        MPI_Comm mpi_comm)
        : DistributedDataParallelBase(module_to_wrap, 0, 1, assigned_device), // Temp rank/world_size
          mpi_communicator_(mpi_comm),
          nccl_communicator_(nullptr), // Initialize to nullptr
          nccl_stream_(nullptr) {

        initialize_distributed_env(mpi_comm); // Sets actual rank_ and world_size_ from MPI

        if (!underlying_module_ptr_) { // This check is in base, but good practice
            throw std::runtime_error("MyFromScratchDDPWithMPIAndNCCL: Module to wrap cannot be null.");
        }

        // Ensure assigned_device (and thus underlying_module_ptr_) is CUDA if using NCCL and not single process
        if (world_size_ > 1 && !device_.is_cuda()) {
            // NCCL primarily works with CUDA devices.
            // If Gloo were an option here, one could switch.
            // For this example, we mandate CUDA for multi-process NCCL.
            std::cerr << "[Rank " << rank_ << "] ERROR: MyFromScratchDDPWithMPIAndNCCL requires a CUDA device for multi-process NCCL operation."
                      << " Provided device: " << device_ << std::endl;
            MPI_Abort(mpi_communicator_, EXIT_FAILURE);
        }

        // Move the underlying module to the final device determined for this rank
        // The base constructor already did this with the initial assigned_device.
        // If initialize_distributed_env changed the effective device, re-affirm.
        underlying_module_ptr_->to(device_);
        module_ref_for_params_ = underlying_module_ptr_; // For clarity that optimizer uses these params

        if (!is_single_process_mode_) {
            setup_nccl_communicator();
            broadcast_initial_parameters();
            MPI_Barrier(mpi_communicator_); // Ensure all setup is done
        }

        std::cout << "[Rank " << rank_ << "] MyFromScratchDDPWithMPIAndNCCL initialized on device " << device_
                  << ". World Size: " << world_size_ << std::endl;
    }

    MyFromScratchDDPWithMPIAndNCCL::~MyFromScratchDDPWithMPIAndNCCL() {
        if (!is_single_process_mode_) {
            if (nccl_communicator_) {
                DDP_NCCLCHECK(ncclCommDestroy(nccl_communicator_));
            }
            if (nccl_stream_) {
                DDP_CUDACHECK(cudaStreamDestroy(nccl_stream_));
            }
        }
        // MPI_Finalize is typically called by the main application, not here.
    }

    void MyFromScratchDDPWithMPIAndNCCL::initialize_distributed_env(MPI_Comm mpi_comm) {
        int mpi_initialized_flag;
        MPI_Initialized(&mpi_initialized_flag);
        if (!mpi_initialized_flag) {
            // This class expects MPI to be initialized by the caller (main function)
            throw std::runtime_error("MPI must be initialized before creating MyFromScratchDDPWithMPIAndNCCL.");
        }

        MPI_Comm_rank(mpi_comm, &rank_); // rank_ is a member of DistributedDataParallelBase
        MPI_Comm_size(mpi_comm, &world_size_); // world_size_ is a member of DistributedDataParallelBase

        is_single_process_mode_ = (world_size_ <= 1);

        if (!is_single_process_mode_) {
            // Determine and set the GPU for this rank
            int num_gpus_on_node = 0;
            if (torch::cuda::is_available()) {
                DDP_CUDACHECK(cudaGetDeviceCount(&num_gpus_on_node));
            }

            if (num_gpus_on_node > 0) {
                // This simple assignment assumes ranks are distributed across nodes
                // such that rank % num_gpus_on_node is a valid local GPU index.
                // More robust would be to get local rank on node.
                int local_gpu_index = rank_ % num_gpus_on_node;
                DDP_CUDACHECK(cudaSetDevice(local_gpu_index));
                device_ = torch::Device(torch::kCUDA, local_gpu_index); // Update device_ from base
            } else {
                 std::cerr << "[Rank " << rank_ << "] Warning: No CUDA GPUs detected for multi-process mode. Defaulting to CPU. NCCL will fail." << std::endl;
                device_ = torch::Device(torch::kCPU); // This will cause issues with NCCL later
            }
        } else { // Single process mode
             if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
                DDP_CUDACHECK(cudaSetDevice(0)); // Default to GPU 0 if available
                device_ = torch::Device(torch::kCUDA, 0);
             } else {
                device_ = torch::Device(torch::kCPU);
             }
        }
        std::cout << "[Rank " << rank_ << "] Effective device set to: " << device_ << std::endl;

    }

    void MyFromScratchDDPWithMPIAndNCCL::setup_nccl_communicator() {
        if (is_single_process_mode_ || !device_.is_cuda()) return; // No NCCL for single process or non-CUDA

        ncclUniqueId nccl_id;
        if (rank_ == 0) {
            DDP_NCCLCHECK(ncclGetUniqueId(&nccl_id));
        }
        // Broadcast the NCCL unique ID from rank 0 to all other processes using MPI
        MPI_Bcast(reinterpret_cast<void*>(&nccl_id), sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_communicator_);

        // Initialize NCCL communicator for each rank
        DDP_NCCLCHECK(ncclCommInitRank(&nccl_communicator_, world_size_, nccl_id, rank_));

        // Create a dedicated CUDA stream for NCCL operations
        DDP_CUDACHECK(cudaStreamCreateWithFlags(&nccl_stream_, cudaStreamNonBlocking));
        std::cout << "[Rank " << rank_ << "] NCCL communicator and stream initialized." << std::endl;
    }

    void MyFromScratchDDPWithMPIAndNCCL::broadcast_initial_parameters() {
        if (is_single_process_mode_ || initial_parameters_broadcasted_) return;

        std::cout << "[Rank " << rank_ << "] Broadcasting initial parameters..." << std::endl;
        torch::NoGradGuard no_grad;
        auto params = module_ref_for_params_->parameters(true);

        for (auto& param : params) {
            if (!param.is_cuda() && device_.is_cuda()) {
                // Should not happen if module_ref_for_params_ was moved to device_ already
                std::cerr << "[Rank " << rank_ << "] Warning: Parameter not on CUDA device during broadcast." << std::endl;
                param.data() = param.data().to(device_); // Ensure it is on the device for NCCL
            }

            ncclDataType_t dtype;
            if (param.scalar_type() == torch::kFloat32) dtype = ncclFloat32; // ncclFloat is for CUDA float
            else if (param.scalar_type() == torch::kFloat16) dtype = ncclFloat16; // ncclHalf
            // Add more type mappings as needed (ncclDouble, ncclInt32 etc.)
            else {
                std::cerr << "[Rank " << rank_ << "] Unsupported parameter DType for NCCL Broadcast: " << param.scalar_type() << std::endl;
                MPI_Abort(mpi_communicator_, EXIT_FAILURE);
            }

            // Rank 0 is the source (root) of the broadcast
            DDP_NCCLCHECK(ncclBroadcast(param.data_ptr(), param.data_ptr(), param.numel(),
                                        dtype, 0, nccl_communicator_, nccl_stream_));
        }
        DDP_CUDACHECK(cudaStreamSynchronize(nccl_stream_)); // Ensure all broadcasts complete
        initial_parameters_broadcasted_ = true;
        std::cout << "[Rank " << rank_ << "] Initial parameters broadcast via NCCL complete." << std::endl;
    }


    std::any MyFromScratchDDPWithMPIAndNCCL::forward(std::initializer_list<std::any> inputs) {
        torch::Tensor data;
        try {
            data = std::any_cast<torch::Tensor>(*inputs.begin());
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("MyFromScratchDDPWithMPIAndNCCL::forward: Input not a torch::Tensor. " + std::string(e.what()));
        }

        // Data should be on this->device_ (ensured by DataLoader or moved here)
        if (data.device() != device_) {
            data = data.to(device_);
        }
        return module_ref_for_params_->forward({data});
    }

    void MyFromScratchDDPWithMPIAndNCCL::synchronize_gradients() {
        if (is_single_process_mode_) {
            return; // No sync needed for single process
        }
        if (!nccl_communicator_) {
            std::cerr << "[Rank " << rank_ << "] ERROR: NCCL communicator not initialized for gradient sync." << std::endl;
            return;
        }

        // std::cout << "[Rank " << rank_ << "] Starting NCCL gradient synchronization..." << std::endl;
        auto params = module_ref_for_params_->parameters(true);

        // NCCL Grouping for potentially better performance (optional for basic implementation)
        // DDP_NCCLCHECK(ncclGroupStart());
        for (const auto& param : params) {
            if (param.grad().defined() && param.grad().numel() > 0) {
                if (!param.grad().is_cuda() && device_.is_cuda()) {
                     std::cerr << "[Rank " << rank_ << "] ERROR: Gradient not on CUDA device for NCCL AllReduce." << std::endl;
                     MPI_Abort(mpi_communicator_, EXIT_FAILURE);
                }

                ncclDataType_t dtype;
                if (param.grad().scalar_type() == torch::kFloat32) dtype = ncclFloat32;
                else if (param.grad().scalar_type() == torch::kFloat16) dtype = ncclFloat16;
                else {
                    std::cerr << "[Rank " << rank_ << "] Unsupported gradient DType for NCCL AllReduce: " << param.grad().scalar_type() << std::endl;
                    MPI_Abort(mpi_communicator_, EXIT_FAILURE);
                }

                // In-place AllReduce: send buffer and receive buffer are the same.
                // Gradients are summed across all ranks.
                DDP_NCCLCHECK(ncclAllReduce(param.grad().data_ptr(), param.grad().data_ptr(),
                                            param.grad().numel(), dtype, ncclSum,
                                            nccl_communicator_, nccl_stream_));
            }
        }
        // DDP_NCCLCHECK(ncclGroupEnd());
        DDP_CUDACHECK(cudaStreamSynchronize(nccl_stream_)); // Wait for all AllReduce operations to complete

        // Average the gradients (they were summed by ncclAllReduce)
        for (auto& param : params) {
            if (param.grad().defined() && param.grad().numel() > 0) {
                param.grad().div_(static_cast<double>(world_size_));
            }
        }
        // std::cout << "[Rank " << rank_ << "] NCCL gradient synchronization complete (summed and averaged)." << std::endl;
    }

    void MyFromScratchDDPWithMPIAndNCCL::to(torch::Device new_device, bool non_blocking) {
        // Moving a DDP module after initialization is complex and generally not recommended
        // without re-initializing the distributed setup (NCCL comms are tied to specific GPUs).
        // This implementation will move the underlying module and mark for re-sync.
        std::cout << "[Rank " << rank_ << "] MyFromScratchDDPWithMPIAndNCCL::to(Device " << new_device
                  << ") called. This is a complex operation for DDP." << std::endl;

        bool device_changed = (new_device != device_);
        DistributedDataParallelBase::to(new_device, non_blocking); // Updates device_ and moves underlying module

        if (device_changed && !is_single_process_mode_) {
            std::cout << "[Rank " << rank_ << "] Device changed. NCCL communicators are now invalid. "
                      << "A full re-initialization of DDP would be needed for multi-GPU operation on new devices." << std::endl;
            // For this example, we'll attempt a parameter re-broadcast, but NCCL comm needs recreation.
            // This simplified `to` is problematic for a running DDP setup.
            // Ideally, DDP is configured for a set of devices and stays there, or is destroyed and recreated.
            if (nccl_communicator_) { // Destroy old NCCL resources
                DDP_NCCLCHECK(ncclCommDestroy(nccl_communicator_));
                nccl_communicator_ = nullptr;
            }
            if (nccl_stream_) {
                DDP_CUDACHECK(cudaStreamDestroy(nccl_stream_));
                nccl_stream_ = nullptr;
            }

            if (device_.is_cuda()){ // Only re-setup NCCL if new device is CUDA
                setup_nccl_communicator(); // Attempt to re-setup NCCL for the new device configuration (complex)
                initial_parameters_broadcasted_ = false;
                broadcast_initial_parameters(); // Re-broadcast parameters
                MPI_Barrier(mpi_communicator_);
            } else {
                 std::cout << "[Rank " << rank_ << "] New device is not CUDA. NCCL cannot be used. DDP effectively disabled." << std::endl;
                 is_single_process_mode_ = true; // Effectively becomes single device now
            }
        }
    }
    // Note: to(ScalarType) and to(Device, ScalarType) would also need similar considerations
    // if they change fundamental properties affecting inter-process consistency or NCCL compatibility.

} // namespace parallel
} // namespace nn
} // namespace xt