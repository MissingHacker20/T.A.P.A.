#include "../Source Code/Training/training.h"
#include "../Source Code/Model/model.h"
#include "../Source Code/NNUE/nnue.h"
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {
bool parse_size(const char* text, std::size_t& value) {
    std::uint64_t parsed = 0;
    const char* end = text;
    while (*end != '\0') ++end;
    const auto result = std::from_chars(text, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end ||
        parsed > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_float(const char* text, float& value) {
    try {
        std::size_t consumed = 0;
        value = std::stof(text, &consumed);
        return text[consumed] == '\0';
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_double(const char* text, double& value) {
    try {
        std::size_t consumed = 0;
        value = std::stod(text, &consumed);
        return text[consumed] == '\0';
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_seed(const char* text, std::uint32_t& value) {
    std::uint64_t parsed = 0;
    const char* end = text;
    while (*end != '\0') ++end;
    const auto result = std::from_chars(text, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end ||
        parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

void print_usage() {
    std::cerr << "Usage: tapa_train <dataset.tsv> <output.nnue> [options]\n"
              << "Options: --epochs N --learning-rate RATE --seed N\n"
              << "         --validation-fraction FRACTION | --validation-file PATH\n"
              << "         --checkpoint-in PATH --checkpoint-out PATH\n"
              << "Legacy positional options remain supported: [epochs] [learning-rate] "
                 "[checkpoint-in] [checkpoint-out]\n";
}
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 2;
    }
    TAPA::Training::Options options;
    int positional_index = 0;
    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.rfind("--", 0) == 0) {
            if (index + 1 >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return 2;
            }
            const char* value = argv[++index];
            if (argument == "--epochs") {
                if (!parse_size(value, options.epochs)) {
                    std::cerr << "Invalid epoch count\n";
                    return 2;
                }
            } else if (argument == "--learning-rate") {
                if (!parse_float(value, options.learning_rate)) {
                    std::cerr << "Invalid learning rate\n";
                    return 2;
                }
            } else if (argument == "--seed") {
                if (!parse_seed(value, options.seed)) {
                    std::cerr << "Invalid seed\n";
                    return 2;
                }
            } else if (argument == "--validation-fraction") {
                if (!parse_double(value, options.validation_fraction)) {
                    std::cerr << "Invalid validation fraction\n";
                    return 2;
                }
            } else if (argument == "--validation-file") {
                options.validation_path = value;
            } else if (argument == "--checkpoint-in") {
                options.checkpoint_in = value;
            } else if (argument == "--checkpoint-out") {
                options.checkpoint_out = value;
            } else {
                std::cerr << "Unknown option: " << argument << '\n';
                print_usage();
                return 2;
            }
        } else {
            switch (positional_index++) {
            case 0:
                if (!parse_size(argument.c_str(), options.epochs)) {
                    std::cerr << "Invalid epoch count\n";
                    return 2;
                }
                break;
            case 1:
                if (!parse_float(argument.c_str(), options.learning_rate)) {
                    std::cerr << "Invalid learning rate\n";
                    return 2;
                }
                break;
            case 2: options.checkpoint_in = argument; break;
            case 3: options.checkpoint_out = argument; break;
            default:
                std::cerr << "Too many positional options\n";
                print_usage();
                return 2;
            }
        }
    }

    std::vector<TAPA::Training::Example> examples;
    std::string error;
    if (!TAPA::Training::load_dataset(argv[1], examples, &error)) {
        std::cerr << "Dataset error: " << error << '\n';
        return 1;
    }
    TAPA::Model::Model model;
    TAPA::Training::Report report;
    if (!TAPA::Training::train(examples, options, model, &report, &error)) {
        std::cerr << "Training error: " << error << '\n';
        return 1;
    }
    if (!TAPA::Model::save(argv[2], model, &error)) {
        std::cerr << "Model save error: " << error << '\n';
        return 1;
    }
    const TAPA::Model::LoadResult saved_model = TAPA::Model::load(argv[2]);
    if (!saved_model) {
        std::cerr << "Saved model reload error: " << saved_model.error << '\n';
        return 1;
    }
    TAPA::NNUE trained_evaluator;
    TAPA::NNUE saved_evaluator;
    if (!trained_evaluator.load_model(model, &error) ||
        !saved_evaluator.load_model(saved_model.model, &error)) {
        std::cerr << "Saved model validation error: " << error << '\n';
        return 1;
    }
    std::int64_t max_reload_error = 0;
    const std::size_t parity_count = std::min<std::size_t>(examples.size(), 64);
    for (std::size_t index = 0; index < parity_count; ++index) {
        const std::int64_t trained_score = trained_evaluator.evaluate(examples[index].position);
        const std::int64_t saved_score = saved_evaluator.evaluate(examples[index].position);
        max_reload_error = std::max(max_reload_error, std::abs(trained_score - saved_score));
    }
    if (max_reload_error != 0) {
        std::cerr << "Saved model prediction mismatch: max absolute delta is "
                  << max_reload_error << " mP\n";
        return 1;
    }
    std::cout << "Training config: seed=" << options.seed
              << ", epochs=" << options.epochs
              << ", learning_rate=" << options.learning_rate
              << ", validation_fraction=" << options.validation_fraction
              << (options.validation_path.empty() ? "" : ", validation_file=" + options.validation_path)
              << (options.checkpoint_in.empty() ? "" : ", resume=" + options.checkpoint_in)
              << (options.checkpoint_out.empty() ? "" : ", checkpoint=" + options.checkpoint_out)
              << "\nQuantization: pawn/piece embedding scale=" << 8.0f
              << " (int16), layer scales=32/128/1000 (int32), activation_limit=32767\n"
              << "Examples: " << examples.size()
              << " (train: " << report.training_examples
              << ", validation: " << report.validation_examples << ")\n"
              << "Initial train MAE: " << report.initial_mae_milli
              << " mP, max error: " << report.initial_max_error_milli << " mP\n";
    for (const auto& metrics : report.epoch_history) {
        std::cout << "Epoch " << metrics.epoch
                  << ": train MAE " << metrics.train_mae_milli
                  << " mP, max error " << metrics.train_max_error_milli << " mP";
        if (report.validation_examples > 0)
            std::cout << ", validation MAE " << metrics.validation_mae_milli
                      << " mP, max error " << metrics.validation_max_error_milli << " mP";
        std::cout << '\n';
    }
    std::cout << "Final train MAE (best checkpoint): " << report.final_mae_milli
              << " mP, max error: " << report.final_max_error_milli << " mP\n";
    if (report.validation_examples > 0)
        std::cout << "Best validation MAE: " << report.final_validation_mae_milli
                  << " mP, max error: " << report.final_validation_max_error_milli
                  << " mP (epoch " << report.best_epoch << ")\n";
    if (report.saturated_parameters > 0)
        std::cout << "WARNING: " << report.saturated_parameters
                  << " parameter(s) were saturated (clamped) during export; "
                  << "the quantized parameters differ from the unbounded training floats\n";
    std::cout << "Max quantized-forward/inference delta: "
              << report.max_export_prediction_error_milli << " mP\n"
              << "Max save/reload prediction delta: " << max_reload_error << " mP\n";
    std::cout << "Saved model: " << argv[2] << '\n';
    return 0;
}
