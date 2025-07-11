#include "include/math/linalg.h" // Include your public header
#include <stdexcept>

// --- Backend Implementation: Eigen ---
// These heavy headers are hidden from the rest of your project.
#include <Eigen/Dense>
#include <Eigen/SVD>

// --- Helper functions for this file only (can be in an anonymous namespace) ---
namespace
{
    // Converts a torch::Tensor to an Eigen::Matrix. This creates a copy.
    Eigen::MatrixXf tensor_to_eigen(const torch::Tensor& tensor)
    {
        if (tensor.dim() != 2)
        {
            throw std::invalid_argument("Input tensor for conversion to Eigen must be 2D.");
        }
        auto tensor_float = tensor.to(torch::kFloat32); // Ensure it's float

        // Eigen::Map lets us view the tensor's data as an Eigen matrix without a copy,
        // but we return a full MatrixXf which does perform a copy. This is safer.
        return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            tensor_float.data_ptr<float>(),
            tensor_float.size(0),
            tensor_float.size(1)
        );
    }

    // Converts an Eigen::Matrix back to a torch::Tensor.
    torch::Tensor eigen_to_tensor(const Eigen::MatrixXf& matrix)
    {
        // We must clone the result because from_blob does not take ownership of the memory.
        // The Eigen matrix will be destroyed when it goes out of scope.
        return torch::from_blob(
            const_cast<float*>(matrix.data()), // from_blob needs non-const, but we know it's safe
            {matrix.rows(), matrix.cols()},
            torch::kFloat32
        ).clone();
    }

    // Converts an Eigen::Vector back to a 1D torch::Tensor.
    torch::Tensor eigen_vector_to_tensor(const Eigen::VectorXf& vec)
    {
        return torch::from_blob(
            const_cast<float*>(vec.data()),
            {vec.size()},
            torch::kFloat32
        ).clone();
    }
} // anonymous namespace


// --- Public API Implementation ---
namespace xt::linalg
{
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd(
        const torch::Tensor& A,
        bool full_matrices)
    {
        // 1. Convert input tensor to an Eigen matrix
        Eigen::MatrixXf eigen_A = tensor_to_eigen(A);

        // 2. Compute SVD using Eigen's robust JacobiSVD or BDCSVD
        // We select the computation options based on the full_matrices flag.
        unsigned int computation_options = full_matrices
                                               ? Eigen::ComputeFullU | Eigen::ComputeFullV
                                               : Eigen::ComputeThinU | Eigen::ComputeThinV;

        Eigen::BDCSVD<Eigen::MatrixXf> svd(eigen_A, computation_options);

        // 3. Extract the results from the Eigen SVD object
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::VectorXf S = svd.singularValues();
        Eigen::MatrixXf Vh = svd.matrixV().transpose();

        // 4. Convert the Eigen results back to torch::Tensors and return them
        return std::make_tuple(
            eigen_to_tensor(U),
            eigen_vector_to_tensor(S),
            eigen_to_tensor(Vh)
        );
    }
} // namespace xt::linalg
