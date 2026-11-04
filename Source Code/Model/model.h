#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include "../Accumulator/pawn_accumulator.h"
#include "../Accumulator/piece_accumulator.h"
#include "../Network/network_model.h"

namespace TAPA::Model
{

    constexpr std::array<char, 8> Magic{'T', 'A', 'P', 'A', 'N', 'N', 'U', 'E'};
    constexpr std::uint32_t FormatVersion = 3;
    constexpr std::uint32_t EndianMarker = 0x01020304u;
    constexpr std::uint32_t InputSize = 512;
    constexpr std::uint32_t Hidden1Size = 32;
    constexpr std::uint32_t Hidden2Size = 32;
    constexpr std::uint32_t OutputSize = 1;
    constexpr std::uint32_t NetworkSchemaVersion = 2; // clipped ReLU and signed linear output

    constexpr std::uint32_t FeatureIndexVersion = 1;
    constexpr std::uint32_t PawnStructureEncodingVersion = 1;

    struct Model
    {
        std::uint32_t input_size = InputSize;
        std::uint32_t hidden1_size = Hidden1Size;
        std::uint32_t hidden2_size = Hidden2Size;
        std::uint32_t output_size = OutputSize;
        std::uint32_t feature_index_version = FeatureIndexVersion;
        std::uint32_t pawn_structure_version = PawnStructureEncodingVersion;
        std::uint32_t network_schema_version = NetworkSchemaVersion;
        Accumulator::PawnWeights pawn_weights{};
        Accumulator::PieceWeights piece_weights{};
        Network::NetworkWeights network_weights{};
    };

    struct LoadResult
    {
        Model model{};
        std::string error;
        explicit operator bool() const noexcept { return error.empty(); }
    };

    LoadResult load(const std::string &path);
    bool save(const std::string &path, const Model &model, std::string *error = nullptr);

} // namespace TAPA::Model
