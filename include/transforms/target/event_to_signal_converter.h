#pragma once

#include "../common.h"


#include <torch/torch.h> // Assumes LibTorch is installed
#include <vector>

namespace xt::transforms::target {

    /**
     * @brief The type of kernel to place at each event location in the signal.
     */
    enum class SignalKernel {
        DIRAC,    // A single point of value 1.0 at the nearest sample.
        GAUSSIAN  // A gaussian bell curve centered on the event time.
    };


    /**
     * @class EventToSignalConverter
     * @brief A target transformation that converts a list of discrete event
     *        timestamps into a continuous signal represented by a fixed-size tensor.
     *
     * This is useful for turning sparse event data into a dense representation
     * suitable for signal processing or convolutional neural networks.
     */
    class EventToSignalConverter : public xt::Module {
    public:
        /**
         * @brief Constructs the EventToSignalConverter.
         *
         * @param signal_duration The total duration of the output signal. The
         *                        timestamps of events should be relative to this.
         * @param num_samples The number of samples (i.e., the length) of the
         *                    output signal tensor.
         * @param kernel The kernel to use for representing events. Defaults to DIRAC.
         * @param gauss_stddev The standard deviation (in number of samples) for
         *                     the Gaussian kernel. Only used if kernel is GAUSSIAN.
         *                     Defaults to 1.0.
         */
        explicit EventToSignalConverter(
            double signal_duration,
            int num_samples,
            SignalKernel kernel = SignalKernel::DIRAC,
            double gauss_stddev = 1.0
        );

        /**
         * @brief Executes the event-to-signal conversion.
         * @param tensors An initializer list expected to contain a single
         *                `std::vector<double>` of event timestamps.
         * @return An std::any containing the generated signal as a 1D `torch::Tensor`
         *         of shape (num_samples,).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double signal_duration_;
        int num_samples_;
        SignalKernel kernel_;
        double gauss_stddev_;

        // Pre-calculated for efficiency
        double samples_per_unit_time_;
    };

} // namespace xt::transforms::target