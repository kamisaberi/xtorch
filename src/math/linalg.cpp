#include <math/linalg.h> // Include your public header
#include <stdexcept>

// --- Backend Implementation: Eigen ---
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues> // Required for eigendecomposition
#include <Eigen/Dense>


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

        return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            tensor_float.data_ptr<float>(),
            tensor_float.size(0),
            tensor_float.size(1)
        );
    }

    // Converts an Eigen::Matrix back to a torch::Tensor.
    torch::Tensor eigen_to_tensor(const Eigen::MatrixXf& matrix)
    {
        return torch::from_blob(
            const_cast<float*>(matrix.data()),
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

    // Helper: Converts a complex Eigen matrix to a complex torch::Tensor.
    torch::Tensor eigen_complex_matrix_to_tensor(const Eigen::MatrixXcf& matrix)
    {
        return torch::from_blob(
                const_cast<std::complex<float>*>(matrix.data()),
                {matrix.rows(), matrix.cols()},
                torch::kComplexFloat
        ).clone();
    }

    // Helper: Converts a complex Eigen vector to a complex 1D torch::Tensor.
    torch::Tensor eigen_complex_vector_to_tensor(const Eigen::VectorXcf& vec)
    {
        return torch::from_blob(
                const_cast<std::complex<float>*>(vec.data()),
                {vec.size()},
                torch::kComplexFloat
        ).clone();
    }

} // anonymous namespace


// --- Public API Implementation ---
namespace xt::linalg
{
    // ... (svd and pinverse implementations are correct) ...
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


    torch::Tensor pinverse(const torch::Tensor& input, double rcond) {
        // --- 1. Input Validation ---
        if (input.dim() != 2) {
            throw std::invalid_argument("Input to pinverse must be a 2D matrix.");
        }
        // Ensure the tensor is on the CPU and is float32. Eigen works best with this.
        auto input_cpu = input.to(torch::kCPU, torch::kFloat32).contiguous();

        long rows = input_cpu.size(0);
        long cols = input_cpu.size(1);
        const float* tensor_ptr = input_cpu.data_ptr<float>();

        // --- 2. Convert torch::Tensor to Eigen::Matrix ---
        // Eigen::Map allows us to create an Eigen matrix that uses the tensor's memory
        // without copying. We specify RowMajor because PyTorch tensors are row-major.
        const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                eigen_matrix(tensor_ptr, rows, cols);

        // --- 3. Compute SVD ---
        // We use JacobiSVD which is robust for this kind of operation.
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(eigen_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // --- 4. Calculate Pseudo-Inverse of Singular Values ---
        auto singular_values = svd.singularValues();
        double tolerance = rcond * singular_values.maxCoeff();

        Eigen::VectorXf singular_values_inv(singular_values.size());
        for (long i = 0; i < singular_values.size(); ++i) {
            if (singular_values(i) > tolerance) {
                singular_values_inv(i) = 1.0f / singular_values(i);
            } else {
                singular_values_inv(i) = 0.0f;
            }
        }

        // --- 5. Compute Pseudo-Inverse Matrix ---
        // A_pinv = V * S_pinv * U^T
        Eigen::MatrixXf result_eigen = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().transpose();

        // --- 6. Convert Eigen::Matrix back to torch::Tensor ---
        // Create an empty tensor with the correct dimensions and options.
        auto output_tensor = torch::empty({result_eigen.rows(), result_eigen.cols()}, input.options());
        float* output_ptr = output_tensor.data_ptr<float>();

        // Map the output tensor's memory and copy the Eigen result into it.
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                output_map(output_ptr, result_eigen.rows(), result_eigen.cols());

        output_map = result_eigen;

        return output_tensor;
    }


    std::tuple<torch::Tensor, torch::Tensor> eig(const torch::Tensor& A)
    {
        // 1. --- Input Validation ---
        if (A.dim() != 2 || A.size(0) != A.size(1))
        {
            throw std::invalid_argument("Input to eig must be a square 2D matrix.");
        }

        // --- START OF FIX ---
        // Convert the input tensor to a REAL Eigen matrix.
        Eigen::MatrixXf eigen_A_real = tensor_to_eigen(A);

        // 2. --- Compute Eigendecomposition using Eigen ---
        // Instantiate EigenSolver with the REAL matrix type. This is the correct usage.
        Eigen::EigenSolver<Eigen::MatrixXf> solver(eigen_A_real, /* computeEigenvectors= */ true);
        // --- END OF FIX ---

        // Check if the computation was successful
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigendecomposition failed to converge.");
        }

        // 3. --- Extract results ---
        // The .eigenvalues() and .eigenvectors() methods correctly return COMPLEX types
        // even though the solver was instantiated with a REAL type.
        Eigen::VectorXcf eigenvalues = solver.eigenvalues();
        Eigen::MatrixXcf eigenvectors = solver.eigenvectors();

        // 4. --- Convert back to torch::Tensors and return ---
        return std::make_tuple(
                eigen_complex_vector_to_tensor(eigenvalues),
                eigen_complex_matrix_to_tensor(eigenvectors)
        );
    }

    //     // --- Implementation of `eig` for GENERAL matrices (returns complex) ---
    // std::tuple<torch::Tensor, torch::Tensor> eig(const torch::Tensor& A)
    // {
    //     if (A.dim() != 2 || A.size(0) != A.size(1)) {
    //         throw std::invalid_argument("Input to eig must be a square 2D matrix.");
    //     }
    //     Eigen::MatrixXf eigen_A_real = tensor_to_eigen(A);
    //     Eigen::EigenSolver<Eigen::MatrixXf> solver(eigen_A_real, /* computeEigenvectors= */ true);
    //
    //     if (solver.info() != Eigen::Success) {
    //         throw std::runtime_error("Eigendecomposition failed to converge.");
    //     }
    //
    //     Eigen::VectorXcf eigenvalues = solver.eigenvalues();
    //     Eigen::MatrixXcf eigenvectors = solver.eigenvectors();
    //
    //     return std::make_tuple(
    //         eigen_complex_vector_to_tensor(eigenvalues),
    //         eigen_complex_matrix_to_tensor(eigenvectors)
    //     );
    // }


    // --- NEW: Correct Implementation of `eigh` for SYMMETRIC matrices (returns real) ---
    std::tuple<torch::Tensor, torch::Tensor> eigh(const torch::Tensor& A, const std::string& UPLO)
    {
        // 1. --- Input Validation ---
        if (A.dim() != 2 || A.size(0) != A.size(1)) {
            throw std::invalid_argument("Input to eigh must be a square 2D matrix.");
        }
        if (UPLO != "U" && UPLO != "L") {
            throw std::invalid_argument("UPLO argument must be 'U' or 'L'.");
        }

        // Convert the input tensor to a REAL Eigen matrix.
        Eigen::MatrixXf eigen_A = tensor_to_eigen(A);

        // 2. --- Compute Eigendecomposition using Eigen's specialized solver ---
        // Use SelfAdjointEigenSolver, which is designed for symmetric matrices.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver;

        // The UPLO flag tells the solver which half of the matrix to read.
        if (UPLO == "U") {
            solver.compute(eigen_A, Eigen::ComputeEigenvectors | Eigen::Upper);
        } else { // UPLO == "L"
            solver.compute(eigen_A, Eigen::ComputeEigenvectors | Eigen::Lower);
        }

        // Check if the computation was successful
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Symmetric eigendecomposition failed to converge.");
        }

        // 3. --- Extract results ---
        // For a symmetric matrix, eigenvalues and eigenvectors are guaranteed to be REAL.
        Eigen::VectorXf eigenvalues = solver.eigenvalues();
        Eigen::MatrixXf eigenvectors = solver.eigenvectors();

        // 4. --- Convert back to REAL torch::Tensors and return ---
        return std::make_tuple(
            eigen_vector_to_tensor(eigenvalues), // Uses the REAL vector helper
            eigen_to_tensor(eigenvectors)        // Uses the REAL matrix helper
        );
    }

} // namespace xt::linalg