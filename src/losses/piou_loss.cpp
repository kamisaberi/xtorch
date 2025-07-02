#include "include/losses/piou_loss.h"

namespace xt::losses
{
    torch::Tensor piou_loss(const torch::Tensor& pred_boxes, const torch::Tensor& gt_boxes, int grid_size = 64)
    {
        // Ensure inputs are valid
        TORCH_CHECK(pred_boxes.dim() == 2 && pred_boxes.size(1) == 5, "Predicted boxes must be 2D (batch_size, 5)");
        TORCH_CHECK(gt_boxes.dim() == 2 && gt_boxes.size(1) == 5, "Ground truth boxes must be 2D (batch_size, 5)");
        TORCH_CHECK(pred_boxes.size(0) == gt_boxes.size(0),
                    "Batch size mismatch between predicted and ground truth boxes");
        TORCH_CHECK(pred_boxes.dtype() == torch::kFloat, "Predicted boxes must be float type");
        TORCH_CHECK(gt_boxes.dtype() == torch::kFloat, "Ground truth boxes must be float type");
        TORCH_CHECK(grid_size > 0, "Grid size must be positive");

        int64_t batch_size = pred_boxes.size(0);
        torch::Tensor loss = torch::zeros({batch_size}, torch::kFloat);

        // Create grid for rasterization
        auto grid = torch::linspace(-1.0f, 1.0f, grid_size, torch::kFloat);
        auto x_grid = grid.view({1, 1, grid_size}).repeat({grid_size, grid_size, 1});
        auto y_grid = grid.view({1, grid_size, 1}).repeat({grid_size, 1, grid_size});
        auto grid_points = torch::stack({x_grid, y_grid}, -1).view({-1, 2}); // Shape: (grid_size*grid_size, 2)

        for (int64_t i = 0; i < batch_size; ++i)
        {
            // Extract box parameters: (x_center, y_center, width, height, angle)
            auto pred_box = pred_boxes[i]; // Shape: (5)
            auto gt_box = gt_boxes[i]; // Shape: (5)

            // Compute rotation matrices
            auto cos_pred = torch::cos(pred_box[4]).item<float>();
            auto sin_pred = torch::sin(pred_box[4]).item<float>();
            auto cos_gt = torch::cos(gt_box[4]).item<float>();
            auto sin_gt = torch::sin(gt_box[4]).item<float>();

            auto pred_rot = torch::tensor({{cos_pred, sin_pred}, {-sin_pred, cos_pred}}, torch::dtype(torch::kFloat));
            auto gt_rot = torch::tensor({{cos_gt, sin_gt}, {-sin_gt, cos_gt}}, torch::dtype(torch::kFloat));

            // Transform grid points to box coordinates
            auto pred_points = torch::matmul(grid_points, pred_rot) + pred_box.slice(0, 0, 2); // Center offset
            auto gt_points = torch::matmul(grid_points, gt_rot) + gt_box.slice(0, 0, 2);

            // Check if points are inside the boxes
            auto pred_inside = (torch::abs(pred_points.select(1, 0)) <= pred_box[2] / 2.0f) &
                (torch::abs(pred_points.select(1, 1)) <= pred_box[3] / 2.0f);
            auto gt_inside = (torch::abs(gt_points.select(1, 0)) <= gt_box[2] / 2.0f) &
                (torch::abs(gt_points.select(1, 1)) <= gt_box[3] / 2.0f);

            // Compute intersection and union
            auto intersection = (pred_inside & gt_inside).to(torch::kFloat).sum();
            auto unon = (pred_inside | gt_inside).to(torch::kFloat).sum();
            auto iou = intersection / (unon + 1e-6f
            );

            // Compute PIoU loss: -log(IoU)
            loss.index_put_({i}, -torch::log(iou + 1e-6f));
        }

        // Return mean loss
        return loss.mean();
    }

    auto PIoULoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::piou_loss(torch::zeros(10), torch::zeros(10));
    }
}
