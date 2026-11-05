#pragma once

#include "../Features/features.h"
#include "../Model/model.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace TAPA::Training {

struct Example {
    Features::Position position{};
    std::int32_t eval_milli = 0;
};

struct Options {
    std::size_t epochs = 10;
    float learning_rate = 0.001f;
    std::uint32_t seed = 0x54415041u;
    // Deterministic train/validation split, derived from `seed`.
    // validation_fraction == 0 disables validation (all examples train).
    double validation_fraction = 0.2;
    // Optional dedicated validation file; when non-empty its examples form the
    // validation set and are never used for weight updates.
    std::string validation_path;
    // Training checkpoint (TODO #13). checkpoint_in: resume from this file if
    // it exists and matches the configuration. checkpoint_out: save the best
    // checkpoint at the end of the run. Checkpoints are training artifacts
    // (float parameters + progress) and are distinct from the exported
    // inference model file written by Model::save().
    std::string checkpoint_in;
    std::string checkpoint_out;
};

struct Report {
    struct EpochMetrics {
        std::size_t epoch = 0;              // 1-based
        float train_mae_milli = 0.0f;
        float train_max_error_milli = 0.0f;
        float validation_mae_milli = 0.0f;  // NaN when no validation set
        float validation_max_error_milli = 0.0f;
    };
    float initial_mae_milli = 0.0f;
    float initial_max_error_milli = 0.0f;
    float final_mae_milli = 0.0f;
    float final_max_error_milli = 0.0f;
    // Best-checkpoint validation MAE in milli-pawns; 0 when no validation set.
    float final_validation_mae_milli = 0.0f;
    float final_validation_max_error_milli = 0.0f;
    std::size_t training_examples = 0;
    std::size_t validation_examples = 0;
    // 1-based epoch whose checkpoint was selected by validation MAE.
    std::size_t best_epoch = 0;
    // Parameters saturated (clamped) by export to the inference format.
    std::size_t saturated_parameters = 0;
    // Maximum absolute score difference between the quantized training
    // forward pass and NNUE inference, over the deterministic parity sample.
    float max_export_prediction_error_milli = 0.0f;
    // Epochs actually completed (early NaN/Inf abort stops the loop).
    std::size_t epochs_run = 0;
    // True when this run started from a training checkpoint.
    bool resumed = false;
    // Per-epoch metrics, in order, for reporting train vs validation MAE.
    std::vector<EpochMetrics> epoch_history;
};

bool parse_fen(const std::string& fen, Features::Position& position,
               bool* white_to_move = nullptr, std::string* error = nullptr);
bool parse_dataset_line(const std::string& line, Example& example,
                        std::string* error = nullptr);
bool load_dataset(const std::string& path, std::vector<Example>& examples,
                  std::string* error = nullptr);
bool train(const std::vector<Example>& examples, const Options& options,
           Model::Model& output, Report* report = nullptr,
           std::string* error = nullptr);

} // namespace TAPA::Training
