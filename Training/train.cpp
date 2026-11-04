#include "../Source Code/Training/training.h"
#include "../Source Code/Model/model.h"
#include <charconv>
#include <cstdint>
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
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: tapa_train <dataset.tsv> <output.nnue> [epochs] [learning-rate]\n";
        return 2;
    }
    TAPA::Training::Options options;
    if (argc >= 4 && !parse_size(argv[3], options.epochs)) {
        std::cerr << "Invalid epoch count\n";
        return 2;
    }
    if (argc >= 5 && !parse_float(argv[4], options.learning_rate)) {
        std::cerr << "Invalid learning rate\n";
        return 2;
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
    std::cout << "Examples: " << examples.size()
              << ", initial MAE: " << report.initial_mae_milli << " mP"
              << ", final MAE: " << report.final_mae_milli << " mP\n"
              << "Saved model: " << argv[2] << '\n';
    return 0;
}
