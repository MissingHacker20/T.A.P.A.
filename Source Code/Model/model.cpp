#include "model.h"
#include <fstream>
#include <vector>
#include <cstdio>
#include <chrono>
#include <thread>
#include "../Network/network.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace TAPA::Model {
namespace {
constexpr std::size_t PayloadSizeValue =
    7 * sizeof(std::uint32_t) +
    64 * static_cast<std::size_t>(Accumulator::PawnNeuronCount) * sizeof(Accumulator::PawnWeight) +
    static_cast<std::size_t>(Accumulator::PieceFeatureCount) * Accumulator::NeuronCount * sizeof(Accumulator::PieceWeight) +
    (static_cast<std::size_t>(Network::InputSize) * Network::HiddenSize1 + Network::HiddenSize1 +
     static_cast<std::size_t>(Network::HiddenSize1) * Network::HiddenSize2 + Network::HiddenSize2 +
     static_cast<std::size_t>(Network::HiddenSize2) * Network::OutputSize + Network::OutputSize + 2 + 2) * sizeof(std::uint32_t);
constexpr std::uint32_t PayloadSize = static_cast<std::uint32_t>(PayloadSizeValue);

void write_u32(std::ostream& out, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) out.put(static_cast<char>((value >> shift) & 0xff));
}

void append_u32(std::vector<std::uint8_t>& payload, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        payload.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
}

void append_i32(std::vector<std::uint8_t>& payload, std::int32_t value) {
    append_u32(payload, static_cast<std::uint32_t>(value));
}

void append_i16(std::vector<std::uint8_t>& payload, std::int16_t value) {
    const auto raw = static_cast<std::uint16_t>(value);
    payload.push_back(static_cast<std::uint8_t>(raw & 0xff));
    payload.push_back(static_cast<std::uint8_t>(raw >> 8));
}

std::int32_t decode_i32(std::uint32_t raw) {
    return raw <= 0x7fffffffu ? static_cast<std::int32_t>(raw)
        : static_cast<std::int32_t>(-1 - static_cast<std::int64_t>(0xffffffffu - raw));
}

std::int16_t decode_i16(std::uint16_t raw) {
    return raw <= 0x7fffu ? static_cast<std::int16_t>(raw)
        : static_cast<std::int16_t>(-1 - static_cast<std::int32_t>(0xffffu - raw));
}

bool read_u32(std::istream& in, std::uint32_t& value) {
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        const int byte = in.get();
        if (byte == std::char_traits<char>::eof()) return false;
        value |= static_cast<std::uint32_t>(byte) << shift;
    }
    return true;
}

std::uint32_t checksum(const std::vector<std::uint8_t>& bytes) {
    std::uint32_t result = 2166136261u;
    for (std::uint8_t byte : bytes) { result ^= byte; result *= 16777619u; }
    return result;
}
}

bool save(const std::string& path, const Model& model, std::string* error) {
    if (path.empty()) { if (error) *error = "empty model path"; return false; }
    if (!model.piece_weights.valid() || model.input_size != InputSize || model.hidden1_size != Hidden1Size ||
        model.hidden2_size != Hidden2Size || model.output_size != OutputSize ||
        model.feature_index_version != FeatureIndexVersion ||
        model.pawn_structure_version != PawnStructureEncodingVersion ||
        model.network_schema_version != NetworkSchemaVersion ||
        model.network_weights.activation_limit <= 0 || model.network_weights.output_scale <= 0) {
        if (error) *error = "incompatible or invalid model architecture";
        return false;
    }
    std::vector<std::uint8_t> payload;
    payload.reserve(PayloadSize);
    append_u32(payload, model.input_size); append_u32(payload, model.hidden1_size);
    append_u32(payload, model.hidden2_size); append_u32(payload, model.output_size);
    append_u32(payload, model.feature_index_version);
    append_u32(payload, model.pawn_structure_version);
    append_u32(payload, model.network_schema_version);
    append_i32(payload, model.pawn_weights.bias);
    for (const auto& row : model.pawn_weights.bit_weights)
        for (Accumulator::PawnWeight value : row) append_i16(payload, value);
    for (const auto& row : model.piece_weights.feature_weights)
        for (Accumulator::PieceWeight value : row) append_i16(payload, value);
    append_i32(payload, model.piece_weights.bias);
    for (const auto& row : model.network_weights.layer1.weights) for (std::int32_t value : row) append_i32(payload, value);
    for (std::int32_t value : model.network_weights.layer1.bias) append_i32(payload, value);
    for (const auto& row : model.network_weights.layer2.weights) for (std::int32_t value : row) append_i32(payload, value);
    for (std::int32_t value : model.network_weights.layer2.bias) append_i32(payload, value);
    for (const auto& row : model.network_weights.output.weights) for (std::int32_t value : row) append_i32(payload, value);
    for (std::int32_t value : model.network_weights.output.bias) append_i32(payload, value);
    append_i32(payload, model.network_weights.activation_limit);
    append_i32(payload, model.network_weights.output_scale);
    if (payload.size() != PayloadSize) {
        if (error) *error = "internal model payload size mismatch";
        return false;
    }
    const std::string temporary_path = path + ".tmp";
    std::ofstream out(temporary_path, std::ios::binary | std::ios::trunc);
    if (!out) { if (error) *error = "cannot open model for writing"; return false; }
    out.write(Magic.data(), Magic.size()); write_u32(out, FormatVersion); write_u32(out, EndianMarker); write_u32(out, Accumulator::PawnNeuronCount);
    write_u32(out, PayloadSize); write_u32(out, checksum(payload));
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    out.flush();
    if (!out) {
        if (error) *error = "cannot write model";
        out.close();
        std::remove(temporary_path.c_str());
        return false;
    }
    out.close();
    // Filesystem watchers (antivirus, cloud sync) may briefly lock freshly written
    // files, so the replace is retried a few times before failing.
    bool replaced = false;
    for (int attempt = 0; attempt < 5 && !replaced; ++attempt) {
#ifdef _WIN32
        replaced = MoveFileExA(temporary_path.c_str(), path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
        if (!replaced) {
            if (std::remove(path.c_str()) == 0)
                replaced = std::rename(temporary_path.c_str(), path.c_str()) == 0;
        }
#else
        replaced = std::rename(temporary_path.c_str(), path.c_str()) == 0;
#endif
        if (!replaced) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!replaced) {
        if (error) *error = "cannot replace model file";
        std::remove(temporary_path.c_str());
        return false;
    }
    return true;
}

LoadResult load(const std::string& path) {
    LoadResult result;
    if (path.empty()) { result.error = "empty model path"; return result; }
    std::ifstream in(path, std::ios::binary);
    if (!in) { result.error = "cannot open model"; return result; }
    std::array<char, 8> magic{}; in.read(magic.data(), magic.size());
    std::uint32_t version = 0, endian = 0, neurons = 0, payload_size = 0, expected_checksum = 0;
    if (!in || magic != Magic || !read_u32(in, version) || !read_u32(in, endian) ||
        !read_u32(in, neurons) || !read_u32(in, payload_size) || !read_u32(in, expected_checksum)) {
        result.error = "invalid model header"; return result;
    }
    if (version != FormatVersion || endian != EndianMarker ||
        neurons != Accumulator::PawnNeuronCount || payload_size != PayloadSize) {
        result.error = "incompatible model architecture"; return result;
    }
    std::vector<std::uint8_t> payload(payload_size);
    in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!in || checksum(payload) != expected_checksum) { result.error = "invalid model checksum or truncated payload"; return result; }
    char trailing = 0;
    if (in.get(trailing)) { result.error = "model contains trailing data"; return result; }
    std::size_t offset = 0;
    auto read_u32_payload = [&payload, &offset]() {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift < 32; shift += 8)
            value |= static_cast<std::uint32_t>(payload[offset++]) << shift;
        return value;
    };
    auto read_i16_payload = [&payload, &offset]() {
        const std::uint16_t raw = static_cast<std::uint16_t>(payload[offset]) |
            (static_cast<std::uint16_t>(payload[offset + 1]) << 8);
        offset += 2;
        return decode_i16(raw);
    };
    result.model.input_size = read_u32_payload(); result.model.hidden1_size = read_u32_payload(); result.model.hidden2_size = read_u32_payload(); result.model.output_size = read_u32_payload();
    if (result.model.input_size != InputSize || result.model.hidden1_size != Hidden1Size ||
        result.model.hidden2_size != Hidden2Size || result.model.output_size != OutputSize ||
        (result.model.feature_index_version = read_u32_payload()) != FeatureIndexVersion ||
        (result.model.pawn_structure_version = read_u32_payload()) != PawnStructureEncodingVersion ||
        (result.model.network_schema_version = read_u32_payload()) != NetworkSchemaVersion) {
        result.error = "incompatible model architecture";
        return result;
    }
    result.model.pawn_weights.bias = decode_i32(read_u32_payload());
    for (auto& row : result.model.pawn_weights.bit_weights)
        for (Accumulator::PawnWeight& value : row) value = read_i16_payload();
    for (auto& row : result.model.piece_weights.feature_weights)
        for (Accumulator::PieceWeight& value : row) value = read_i16_payload();
    result.model.piece_weights.bias = decode_i32(read_u32_payload());
    for (auto& row : result.model.network_weights.layer1.weights)
        for (std::int32_t& value : row) value = decode_i32(read_u32_payload());
    for (std::int32_t& value : result.model.network_weights.layer1.bias) value = decode_i32(read_u32_payload());
    for (auto& row : result.model.network_weights.layer2.weights)
        for (std::int32_t& value : row) value = decode_i32(read_u32_payload());
    for (std::int32_t& value : result.model.network_weights.layer2.bias) value = decode_i32(read_u32_payload());
    for (auto& row : result.model.network_weights.output.weights)
        for (std::int32_t& value : row) value = decode_i32(read_u32_payload());
    for (std::int32_t& value : result.model.network_weights.output.bias) value = decode_i32(read_u32_payload());
    result.model.network_weights.activation_limit = decode_i32(read_u32_payload());
    result.model.network_weights.output_scale = decode_i32(read_u32_payload());
    if (result.model.network_weights.activation_limit <= 0 || result.model.network_weights.output_scale <= 0) {
        result.error = "invalid network activation limit or output scale";
        return result;
    }
    return result;
}

} // namespace TAPA::Model
