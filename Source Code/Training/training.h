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
};

struct Report {
    float initial_mae_milli = 0.0f;
    float final_mae_milli = 0.0f;
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
