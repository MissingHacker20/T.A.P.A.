#include "training.h"
#include "../Network/network.h"
#include "../NNUE/nnue.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string_view>

namespace TAPA::Training {
namespace {

constexpr float PawnEmbeddingScale = 8.0f;
constexpr float PieceEmbeddingScale = 8.0f;
constexpr float Hidden1Scale = 32.0f;
constexpr float Hidden2Scale = 128.0f;
constexpr float ScoreScale = 1000.0f;
constexpr std::int32_t ActivationLimit = 32767;
constexpr float HuberDelta = 2.0f;

// Representable ranges of the exported inference format. Training-side
// quantization must saturate exactly like to_i16()/to_i32() at export time,
// otherwise a prediction made during training can differ from the prediction
// made by the exported model for the same parameters.
constexpr double I16MinUnits = -32768.0;
constexpr double I16MaxUnits = 32767.0;
constexpr double I32MinUnits = -2147483648.0;
constexpr double I32MaxUnits = 2147483647.0;

using Vector256 = std::array<float, Accumulator::NeuronCount>;
using InputVector = std::array<float, Network::InputSize>;
using Hidden1 = std::array<float, Network::HiddenSize1>;
using Hidden2 = std::array<float, Network::HiddenSize2>;

template<int In, int Out>
struct FloatLayer {
    std::array<std::array<float, Out>, In> weights{};
    std::array<float, Out> bias{};
};

struct FloatModel {
    // Single shared embedding bias (TODO #12): pawn and piece contributions
    // used to carry two independent bias parameters with identical gradients.
    // Both are now driven by this one value; the exported model still stores
    // both bias fields (same value) so the inference format is unchanged.
    float embedding_bias = 0.0f;
    std::array<Vector256, 64> pawn_weights{};
    std::vector<Vector256> piece_weights;
    FloatLayer<Network::InputSize, Network::HiddenSize1> layer1{};
    FloatLayer<Network::HiddenSize1, Network::HiddenSize2> layer2{};
    FloatLayer<Network::HiddenSize2, Network::OutputSize> output{};

    FloatModel() : piece_weights(Accumulator::PieceFeatureCount) {}
};

struct ForwardState {
    InputVector input{};
    Hidden1 hidden1{};
    Hidden2 hidden2{};
    float prediction = 0.0f;
};

// Quantize to the inference grid and saturate to the representable integer
// range of the exported parameter, mirroring to_i16()/to_i32() semantics.
// min_units/max_units are expressed in exported integer units (i.e. after
// multiplying by `scale`).
float quantize(float value, float scale, double min_units, double max_units) noexcept {
    const double scaled = std::clamp(static_cast<double>(value) * scale, min_units, max_units);
    return static_cast<float>(std::round(scaled) / scale);
}

// Same rounding as quantize() but for exported layer weights/biases whose
// products are stored as int32_t.
constexpr double LayerMinUnits = I32MinUnits;
constexpr double LayerMaxUnits = I32MaxUnits;

float clipped_relu(float value, float limit) noexcept {
    return std::clamp(value, 0.0f, limit);
}

template<int In, int Out>
void initialize_layer(FloatLayer<In, Out>& layer, std::mt19937& random, float standard_deviation) {
    std::normal_distribution<float> distribution(0.0f, standard_deviation);
    for (auto& row : layer.weights)
        for (float& weight : row) weight = distribution(random);
}

FloatModel make_initial_model(std::mt19937& random) {
    FloatModel model;
    std::normal_distribution<float> embedding_distribution(0.0f, 0.2f);
    for (auto& row : model.pawn_weights)
        for (float& weight : row) weight = embedding_distribution(random);
    for (auto& row : model.piece_weights)
        for (float& weight : row) weight = embedding_distribution(random);

    initialize_layer(model.layer1, random, 0.08f);
    initialize_layer(model.layer2, random, 0.1f);
    initialize_layer(model.output, random, 0.2f);
    return model;
}

void build_inputs(const Example& example, const FloatModel& model, InputVector& input) noexcept {
    Vector256 pawn{};
    Vector256 piece_white{};
    Vector256 piece_black{};
    for (std::size_t neuron = 0; neuron < pawn.size(); ++neuron)
        pawn[neuron] = quantize(model.embedding_bias, PawnEmbeddingScale, I32MinUnits, I32MaxUnits);

    const Features::PawnStructureKey key = Features::make_pawn_structure_key(example.position);
    for (std::size_t bit = 0; bit < 64; ++bit) {
        if ((key & (Features::PawnStructureKey{1} << bit)) == 0) continue;
        for (std::size_t neuron = 0; neuron < pawn.size(); ++neuron)
            pawn[neuron] += quantize(model.pawn_weights[bit][neuron], PawnEmbeddingScale, I16MinUnits, I16MaxUnits);
    }

    Features::FeatureSet features;
    Features::generate_features(example.position, features);
    for (std::size_t feature_index = 0; feature_index < features.size; ++feature_index) {
        const Features::Feature& feature = features.features[feature_index];
        const Vector256& row = model.piece_weights[feature.index];
        Vector256& destination = feature.perspective == Features::PieceColor::White ? piece_white : piece_black;
        for (std::size_t neuron = 0; neuron < destination.size(); ++neuron)
            destination[neuron] += quantize(row[neuron], PieceEmbeddingScale, I16MinUnits, I16MaxUnits);
    }

    for (std::size_t neuron = 0; neuron < pawn.size(); ++neuron) {
        const float common = pawn[neuron] + quantize(model.embedding_bias, PieceEmbeddingScale, I32MinUnits, I32MaxUnits);
        input[neuron] = common + piece_white[neuron];
        input[neuron + Accumulator::NeuronCount] = common + piece_black[neuron];
    }
}

template<int In, int Out>
float quantized_weight(const FloatLayer<In, Out>& layer, std::size_t input, std::size_t output,
                       float scale) noexcept {
    return quantize(layer.weights[input][output], scale,
                    LayerMinUnits, LayerMaxUnits);
}

ForwardState forward(const Example& example, const FloatModel& model) noexcept {
    ForwardState state;
    build_inputs(example, model, state.input);

    constexpr float hidden1_limit = static_cast<float>(ActivationLimit) / Hidden1Scale;
    constexpr float hidden2_limit = static_cast<float>(ActivationLimit) / Hidden2Scale;
    for (int output = 0; output < Network::HiddenSize1; ++output) {
        float sum = quantize(model.layer1.bias[static_cast<std::size_t>(output)], Hidden1Scale, LayerMinUnits, LayerMaxUnits);
        for (int input = 0; input < Network::InputSize; ++input)
            sum += state.input[static_cast<std::size_t>(input)] *
                quantized_weight(model.layer1, input, output, Hidden1Scale / PieceEmbeddingScale);
        state.hidden1[static_cast<std::size_t>(output)] = clipped_relu(sum, hidden1_limit);
    }

    for (int output = 0; output < Network::HiddenSize2; ++output) {
        float sum = quantize(model.layer2.bias[static_cast<std::size_t>(output)], Hidden2Scale, LayerMinUnits, LayerMaxUnits);
        for (int input = 0; input < Network::HiddenSize1; ++input)
            sum += state.hidden1[static_cast<std::size_t>(input)] *
                quantized_weight(model.layer2, input, output, Hidden2Scale / Hidden1Scale);
        state.hidden2[static_cast<std::size_t>(output)] = clipped_relu(sum, hidden2_limit);
    }

    state.prediction = quantize(model.output.bias[0], ScoreScale, LayerMinUnits, LayerMaxUnits);
    for (int input = 0; input < Network::HiddenSize2; ++input)
        state.prediction += state.hidden2[static_cast<std::size_t>(input)] *
            quantized_weight(model.output, input, 0, ScoreScale / Hidden2Scale);
    return state;
}

float huber_gradient(float residual) noexcept {
    return std::clamp(residual, -HuberDelta, HuberDelta);
}

void train_one(const Example& example, FloatModel& model, float learning_rate) noexcept {
    const ForwardState state = forward(example, model);
    const float target = static_cast<float>(example.eval_milli) / ScoreScale;
    const float output_gradient = huber_gradient(state.prediction - target);

    Hidden2 hidden2_gradient{};
    for (int input = 0; input < Network::HiddenSize2; ++input) {
        const std::size_t index = static_cast<std::size_t>(input);
        const float weight = quantized_weight(model.output, input, 0, ScoreScale / Hidden2Scale);
        hidden2_gradient[index] = output_gradient * weight;
        model.output.weights[index][0] -= learning_rate * output_gradient * state.hidden2[index];
    }
    model.output.bias[0] -= learning_rate * output_gradient;

    Hidden2 hidden2_preactivation_gradient{};
    for (int output = 0; output < Network::HiddenSize2; ++output) {
        const std::size_t index = static_cast<std::size_t>(output);
        if (state.hidden2[index] > 0.0f &&
            state.hidden2[index] < static_cast<float>(ActivationLimit) / Hidden2Scale)
            hidden2_preactivation_gradient[index] = hidden2_gradient[index];
    }

    Hidden1 hidden1_gradient{};
    for (int input = 0; input < Network::HiddenSize1; ++input) {
        const std::size_t index = static_cast<std::size_t>(input);
        float gradient = 0.0f;
        for (int output = 0; output < Network::HiddenSize2; ++output) {
            const std::size_t output_index = static_cast<std::size_t>(output);
            gradient += hidden2_preactivation_gradient[output_index] *
                quantized_weight(model.layer2, input, output, Hidden2Scale / Hidden1Scale);
        }
        hidden1_gradient[index] = gradient;
    }

    Hidden1 hidden1_preactivation_gradient{};
    for (int index = 0; index < Network::HiddenSize1; ++index) {
        const std::size_t neuron = static_cast<std::size_t>(index);
        if (state.hidden1[neuron] > 0.0f &&
            state.hidden1[neuron] < static_cast<float>(ActivationLimit) / Hidden1Scale)
            hidden1_preactivation_gradient[neuron] = hidden1_gradient[neuron];
    }

    InputVector input_gradient{};
    std::size_t input = 0;
    for (const auto& weights : model.layer1.weights) {
        float gradient = 0.0f;
        for (std::size_t output = 0; output < hidden1_preactivation_gradient.size(); ++output) {
            gradient += hidden1_preactivation_gradient[output] *
                quantize(weights[output], Hidden1Scale / PieceEmbeddingScale,
                         LayerMinUnits, LayerMaxUnits);
        }
        input_gradient[input] = gradient;
        ++input;
    }

    for (int input = 0; input < Network::HiddenSize1; ++input) {
        const std::size_t input_index = static_cast<std::size_t>(input);
        for (int output = 0; output < Network::HiddenSize2; ++output) {
            const std::size_t output_index = static_cast<std::size_t>(output);
            model.layer2.weights[input_index][output_index] -=
                learning_rate * hidden2_preactivation_gradient[output_index] * state.hidden1[input_index];
        }
    }
    for (int output = 0; output < Network::HiddenSize2; ++output) {
        const std::size_t index = static_cast<std::size_t>(output);
        model.layer2.bias[index] -= learning_rate * hidden2_preactivation_gradient[index];
    }
    for (int input = 0; input < Network::InputSize; ++input) {
        const std::size_t input_index = static_cast<std::size_t>(input);
        for (int output = 0; output < Network::HiddenSize1; ++output) {
            const std::size_t output_index = static_cast<std::size_t>(output);
            model.layer1.weights[input_index][output_index] -=
                learning_rate * hidden1_preactivation_gradient[output_index] * state.input[input_index];
        }
    }
    for (int output = 0; output < Network::HiddenSize1; ++output) {
        const std::size_t index = static_cast<std::size_t>(output);
        model.layer1.bias[index] -= learning_rate * hidden1_preactivation_gradient[index];
    }

    Vector256 pawn_gradient{};
    Vector256 piece_white_gradient{};
    Vector256 piece_black_gradient{};
    for (std::size_t neuron = 0; neuron < Accumulator::NeuronCount; ++neuron) {
        piece_white_gradient[neuron] = input_gradient[neuron];
        piece_black_gradient[neuron] = input_gradient[neuron + Accumulator::NeuronCount];
        pawn_gradient[neuron] = piece_white_gradient[neuron] + piece_black_gradient[neuron];
    }
    float pawn_bias_gradient = 0.0f;
    float piece_bias_gradient = 0.0f;
    for (std::size_t neuron = 0; neuron < Accumulator::NeuronCount; ++neuron) {
        pawn_bias_gradient += pawn_gradient[neuron];
        piece_bias_gradient += piece_white_gradient[neuron] + piece_black_gradient[neuron];
    }
    // Single shared bias: gradients from both perspectives sum into one update.
    model.embedding_bias -= learning_rate * (pawn_bias_gradient + piece_bias_gradient);

    const Features::PawnStructureKey key = Features::make_pawn_structure_key(example.position);
    for (std::size_t bit = 0; bit < 64; ++bit) {
        if ((key & (Features::PawnStructureKey{1} << bit)) == 0) continue;
        for (std::size_t neuron = 0; neuron < Accumulator::NeuronCount; ++neuron)
            model.pawn_weights[bit][neuron] -= learning_rate * pawn_gradient[neuron];
    }

    Features::FeatureSet features;
    Features::generate_features(example.position, features);
    for (std::size_t feature_index = 0; feature_index < features.size; ++feature_index) {
        const Features::Feature& feature = features.features[feature_index];
        const Vector256& gradient = feature.perspective == Features::PieceColor::White
            ? piece_white_gradient : piece_black_gradient;
        Vector256& row = model.piece_weights[feature.index];
        for (std::size_t neuron = 0; neuron < Accumulator::NeuronCount; ++neuron)
            row[neuron] -= learning_rate * gradient[neuron];
    }
}

struct ErrorMetrics {
    float mean_absolute_error_milli = 0.0f;
    float max_absolute_error_milli = 0.0f;
};

ErrorMetrics error_metrics(const std::vector<Example>& examples, const FloatModel& model) noexcept {
    double absolute_error = 0.0;
    float max_absolute_error = 0.0f;
    for (const Example& example : examples) {
        const float target = static_cast<float>(example.eval_milli) / ScoreScale;
        const float error = std::abs(forward(example, model).prediction - target);
        absolute_error += error;
        max_absolute_error = std::max(max_absolute_error, error);
    }
    return {
        static_cast<float>(absolute_error * (ScoreScale / static_cast<double>(examples.size()))),
        max_absolute_error * ScoreScale
    };
}

bool finite_parameters(const FloatModel& model) noexcept {
    auto finite_range = [](const auto& range) {
        for (const auto& value : range)
            if (!std::isfinite(value)) return false;
        return true;
    };
    if (!std::isfinite(model.embedding_bias)) return false;
    for (const Vector256& row : model.pawn_weights)
        if (!finite_range(row)) return false;
    for (const Vector256& row : model.piece_weights)
        if (!finite_range(row)) return false;
    for (const auto& row : model.layer1.weights)
        if (!finite_range(row)) return false;
    for (const auto& row : model.layer2.weights)
        if (!finite_range(row)) return false;
    for (const auto& row : model.output.weights)
        if (!finite_range(row)) return false;
    return finite_range(model.layer1.bias) && finite_range(model.layer2.bias) &&
        finite_range(model.output.bias);
}

std::int32_t to_i32(float value) noexcept {
    const double rounded = std::round(static_cast<double>(value));
    return static_cast<std::int32_t>(std::clamp(rounded,
        static_cast<double>(std::numeric_limits<std::int32_t>::min()),
        static_cast<double>(std::numeric_limits<std::int32_t>::max())));
}

std::int16_t to_i16(float value) noexcept {
    const double rounded = std::round(static_cast<double>(value));
    return static_cast<std::int16_t>(std::clamp(rounded,
        static_cast<double>(std::numeric_limits<std::int16_t>::min()),
        static_cast<double>(std::numeric_limits<std::int16_t>::max())));
}

// Saturation-aware export helpers: each returns the exported value and
// increments `saturated` when the parameter changed because it did not fit
// the representable range of the target integer type.
struct ExportSaturated {
    std::size_t& saturated;
    std::int16_t i16(float scaled_value) noexcept {
        const double rounded = std::round(static_cast<double>(scaled_value));
        if (rounded < I16MinUnits || rounded > I16MaxUnits) ++saturated;
        return to_i16(scaled_value);
    }
    std::int32_t i32(float scaled_value) noexcept {
        const double rounded = std::round(static_cast<double>(scaled_value));
        if (rounded < I32MinUnits || rounded > I32MaxUnits) ++saturated;
        return to_i32(scaled_value);
    }
};

// --- Training checkpoint (TODO #13) -------------------------------------
// Checkpoints store float training parameters plus progress metadata and are
// a training-only artifact, distinct from the exported inference model.
// Layout: raw host-endian POD values; not portable across platforms.
constexpr char CheckpointMagic[8] = {'T','A','P','A','C','H','K','P'};
constexpr std::uint32_t CheckpointVersion = 2;

struct CheckpointState {
    std::uint32_t seed = 0;
    float learning_rate = 0.0f;
    std::uint64_t epochs_requested = 0;
    double validation_fraction = 0.0;
    std::uint64_t training_count = 0;
    std::uint64_t validation_count = 0;
    std::uint64_t dataset_signature = 0;
    std::uint64_t epochs_done = 0;
    float best_validation_mae = 0.0f;
    bool had_validation = false;
    std::uint64_t best_epoch = 0;
    FloatModel model;
    FloatModel best_model;
};

void hash_checkpoint_value(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        hash ^= static_cast<std::uint8_t>(value >> shift);
        hash *= 1099511628211ULL;
    }
}

void hash_examples(std::uint64_t& hash, const std::vector<Example>& examples) noexcept {
    hash_checkpoint_value(hash, examples.size());
    for (const Example& example : examples) {
        hash_checkpoint_value(hash, static_cast<std::uint32_t>(example.eval_milli));
        hash_checkpoint_value(hash, example.position.num_pieces);
        for (std::size_t index = 0; index < example.position.num_pieces; ++index) {
            const Features::Piece& piece = example.position.pieces[index];
            hash_checkpoint_value(hash, static_cast<std::uint8_t>(piece.type));
            hash_checkpoint_value(hash, static_cast<std::uint8_t>(piece.color));
            hash_checkpoint_value(hash, static_cast<std::uint32_t>(piece.square));
        }
    }
}

void hash_model(std::uint64_t& hash, const FloatModel& model) noexcept {
    auto add_float = [&hash](float value) {
        hash_checkpoint_value(hash, std::bit_cast<std::uint32_t>(value));
    };
    auto add_range = [&add_float](const auto& range) {
        for (const auto& value : range) add_float(value);
    };
    add_float(model.embedding_bias);
    for (const Vector256& row : model.pawn_weights) add_range(row);
    for (const Vector256& row : model.piece_weights) add_range(row);
    for (const auto& row : model.layer1.weights) add_range(row);
    add_range(model.layer1.bias);
    for (const auto& row : model.layer2.weights) add_range(row);
    add_range(model.layer2.bias);
    for (const auto& row : model.output.weights) add_range(row);
    add_range(model.output.bias);
}

std::uint64_t dataset_signature(const std::vector<Example>& training,
                                const std::vector<Example>& validation) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    hash_examples(hash, training);
    hash_examples(hash, validation);
    return hash;
}

std::uint64_t checkpoint_checksum(const CheckpointState& state) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    hash_checkpoint_value(hash, state.seed);
    hash_checkpoint_value(hash, std::bit_cast<std::uint32_t>(state.learning_rate));
    hash_checkpoint_value(hash, state.epochs_requested);
    hash_checkpoint_value(hash, std::bit_cast<std::uint64_t>(state.validation_fraction));
    hash_checkpoint_value(hash, state.training_count);
    hash_checkpoint_value(hash, state.validation_count);
    hash_checkpoint_value(hash, state.dataset_signature);
    hash_checkpoint_value(hash, state.epochs_done);
    hash_checkpoint_value(hash, std::bit_cast<std::uint32_t>(state.best_validation_mae));
    hash_checkpoint_value(hash, state.had_validation ? 1 : 0);
    hash_checkpoint_value(hash, state.best_epoch);
    hash_model(hash, state.model);
    hash_model(hash, state.best_model);
    return hash;
}

template<typename T>
bool write_pod(std::ostream& out, const T& value, std::string* error) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        if (error) *error = "failed to write training checkpoint";
        return false;
    }
    return true;
}

template<typename T>
bool read_pod(std::istream& in, T& value, std::string* error) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        if (error) *error = "training checkpoint is truncated or corrupt";
        return false;
    }
    return true;
}

bool write_checkpoint_weights(std::ostream& out, const FloatModel& model, std::string* error) {
    if (!write_pod(out, model.embedding_bias, error)) return false;
    for (const Vector256& row : model.pawn_weights)
        if (!write_pod(out, row, error)) return false;
    if (model.piece_weights.size() != Accumulator::PieceFeatureCount) {
        if (error) *error = "internal error: piece weights table size mismatch";
        return false;
    }
    if (!out.write(reinterpret_cast<const char*>(model.piece_weights.data()),
            static_cast<std::streamsize>(model.piece_weights.size() * sizeof(Vector256)))) {
        if (error) *error = "failed to write training checkpoint";
        return false;
    }
    for (const auto& row : model.layer1.weights)
        if (!write_pod(out, row, error)) return false;
    if (!write_pod(out, model.layer1.bias, error)) return false;
    for (const auto& row : model.layer2.weights)
        if (!write_pod(out, row, error)) return false;
    if (!write_pod(out, model.layer2.bias, error)) return false;
    if (!write_pod(out, model.output.weights, error)) return false;
    return write_pod(out, model.output.bias, error);
}

bool read_checkpoint_weights(std::istream& in, FloatModel& model, std::string* error) {
    if (!read_pod(in, model.embedding_bias, error)) return false;
    for (Vector256& row : model.pawn_weights)
        if (!read_pod(in, row, error)) return false;
    if (model.piece_weights.size() != Accumulator::PieceFeatureCount) {
        if (error) *error = "internal error: piece weights table size mismatch";
        return false;
    }
    if (!in.read(reinterpret_cast<char*>(model.piece_weights.data()),
            static_cast<std::streamsize>(model.piece_weights.size() * sizeof(Vector256)))) {
        if (error) *error = "training checkpoint is truncated or corrupt";
        return false;
    }
    for (auto& row : model.layer1.weights)
        if (!read_pod(in, row, error)) return false;
    if (!read_pod(in, model.layer1.bias, error)) return false;
    for (auto& row : model.layer2.weights)
        if (!read_pod(in, row, error)) return false;
    if (!read_pod(in, model.layer2.bias, error)) return false;
    if (!read_pod(in, model.output.weights, error)) return false;
    return read_pod(in, model.output.bias, error);
}

bool save_checkpoint(const std::string& path, const CheckpointState& state, std::string* error) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (error) *error = "cannot open checkpoint file for writing: " + path;
        return false;
    }
    if (!out.write(CheckpointMagic, 8) || !write_pod(out, CheckpointVersion, error)) {
        if (error && error->empty()) *error = "failed to write training checkpoint";
        return false;
    }
    if (!write_pod(out, state.seed, error) ||
        !write_pod(out, state.learning_rate, error) ||
        !write_pod(out, state.epochs_requested, error) ||
        !write_pod(out, state.validation_fraction, error) ||
        !write_pod(out, state.training_count, error) ||
        !write_pod(out, state.validation_count, error) ||
        !write_pod(out, state.dataset_signature, error) ||
        !write_pod(out, state.epochs_done, error) ||
        !write_pod(out, state.best_validation_mae, error) ||
        !write_pod(out, state.had_validation, error) ||
        !write_pod(out, state.best_epoch, error)) return false;
    if (!write_checkpoint_weights(out, state.model, error) ||
        !write_checkpoint_weights(out, state.best_model, error) ||
        !write_pod(out, checkpoint_checksum(state), error)) return false;
    out.flush();
    if (!out) {
        if (error) *error = "failed to flush training checkpoint";
        return false;
    }
    if (error) error->clear();
    return true;
}

bool load_checkpoint(const std::string& path, CheckpointState& state, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) *error = "cannot open checkpoint file for reading: " + path;
        return false;
    }
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, CheckpointMagic, 8) != 0) {
        if (error) *error = "file is not a T.A.P.A. training checkpoint: " + path;
        return false;
    }
    std::uint32_t version = 0;
    if (!read_pod(in, version, error)) return false;
    if (version != CheckpointVersion) {
        if (error) *error = "unsupported training checkpoint version " + std::to_string(version);
        return false;
    }
    if (!read_pod(in, state.seed, error) ||
        !read_pod(in, state.learning_rate, error) ||
        !read_pod(in, state.epochs_requested, error) ||
        !read_pod(in, state.validation_fraction, error) ||
        !read_pod(in, state.training_count, error) ||
        !read_pod(in, state.validation_count, error) ||
        !read_pod(in, state.dataset_signature, error) ||
        !read_pod(in, state.epochs_done, error) ||
        !read_pod(in, state.best_validation_mae, error) ||
        !read_pod(in, state.had_validation, error) ||
        !read_pod(in, state.best_epoch, error)) return false;
    if (!read_checkpoint_weights(in, state.model, error) ||
        !read_checkpoint_weights(in, state.best_model, error)) return false;
    std::uint64_t stored_checksum = 0;
    if (!read_pod(in, stored_checksum, error)) return false;
    if (in.peek() != std::char_traits<char>::eof()) {
        if (error) *error = "training checkpoint has unexpected trailing data";
        return false;
    }
    if (stored_checksum != checkpoint_checksum(state)) {
        if (error) *error = "training checkpoint checksum mismatch";
        return false;
    }
    if (error) error->clear();
    return true;
}

template<int In, int Out>
void export_layer(const FloatLayer<In, Out>& source,
                  Network::Layer<In, Out>& destination, float weight_scale, float bias_scale,
                  ExportSaturated& saturated) noexcept {
    for (int input = 0; input < In; ++input)
        for (int output = 0; output < Out; ++output)
            destination.weights[static_cast<std::size_t>(input)][static_cast<std::size_t>(output)] =
                saturated.i32(source.weights[static_cast<std::size_t>(input)][static_cast<std::size_t>(output)] * weight_scale);
    for (int output = 0; output < Out; ++output)
        destination.bias[static_cast<std::size_t>(output)] =
            saturated.i32(source.bias[static_cast<std::size_t>(output)] * bias_scale);
}

std::size_t export_model(const FloatModel& source, Model::Model& destination) noexcept {
    std::size_t saturated = 0;
    ExportSaturated saturating{saturated};
    destination.pawn_weights.bias = saturating.i32(source.embedding_bias * PawnEmbeddingScale);
    destination.piece_weights.bias = saturating.i32(source.embedding_bias * PieceEmbeddingScale);
    for (std::size_t neuron = 0; neuron < Accumulator::NeuronCount; ++neuron) {
        for (std::size_t bit = 0; bit < 64; ++bit)
            destination.pawn_weights.bit_weights[bit][neuron] =
                saturating.i16(source.pawn_weights[bit][neuron] * PawnEmbeddingScale);
        for (std::size_t feature = 0; feature < Accumulator::PieceFeatureCount; ++feature)
            destination.piece_weights.feature_weights[feature][neuron] =
                saturating.i16(source.piece_weights[feature][neuron] * PieceEmbeddingScale);
    }
    export_layer(source.layer1, destination.network_weights.layer1,
                 Hidden1Scale / PieceEmbeddingScale, Hidden1Scale, saturating);
    export_layer(source.layer2, destination.network_weights.layer2,
                 Hidden2Scale / Hidden1Scale, Hidden2Scale, saturating);
    export_layer(source.output, destination.network_weights.output,
                 ScoreScale / Hidden2Scale, ScoreScale, saturating);
    destination.network_weights.activation_limit = ActivationLimit;
    destination.network_weights.output_scale = 1;
    return saturated;
}

bool parse_integer(std::string_view text, std::int64_t& value) noexcept {
    if (text.empty()) return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool parse_fen_board(std::string_view board, Features::Position& position) noexcept {
    int rank = 7;
    int file = 0;
    bool previous_was_digit = false;
    for (char symbol : board) {
        if (symbol == '/') {
            if (file != 8 || rank == 0) return false;
            --rank;
            file = 0;
            previous_was_digit = false;
            continue;
        }
        if (symbol >= '1' && symbol <= '8') {
            if (previous_was_digit) return false;
            file += symbol - '0';
            if (file > 8) return false;
            previous_was_digit = true;
            continue;
        }
        Features::PieceType type;
        switch (symbol) {
        case 'p': case 'P': type = Features::PieceType::Pawn; break;
        case 'n': case 'N': type = Features::PieceType::Knight; break;
        case 'b': case 'B': type = Features::PieceType::Bishop; break;
        case 'r': case 'R': type = Features::PieceType::Rook; break;
        case 'q': case 'Q': type = Features::PieceType::Queen; break;
        case 'k': case 'K': type = Features::PieceType::King; break;
        default: return false;
        }
        if (file >= 8 || position.num_pieces >= std::size(position.pieces)) return false;
        const Features::PieceColor color = symbol >= 'A' && symbol <= 'Z'
            ? Features::PieceColor::White : Features::PieceColor::Black;
        position.pieces[position.num_pieces++] = {type, color, rank * 8 + file};
        ++file;
        previous_was_digit = false;
    }
    return rank == 0 && file == 8;
}

bool valid_castling(std::string_view field) noexcept {
    if (field == "-") return true;
    if (field.empty()) return false;
    std::array<bool, 4> seen{};
    for (char symbol : field) {
        std::size_t index;
        switch (symbol) {
        case 'K': index = 0; break;
        case 'Q': index = 1; break;
        case 'k': index = 2; break;
        case 'q': index = 3; break;
        default: return false;
        }
        if (seen[index]) return false;
        seen[index] = true;
    }
    return true;
}

} // namespace

bool parse_fen(const std::string& fen, Features::Position& position,
               bool* white_to_move, std::string* error) {
    auto fail = [error](const char* message) {
        if (error) *error = message;
        return false;
    };
    std::array<std::string_view, 6> fields{};
    std::size_t field_count = 0;
    std::size_t cursor = 0;
    while (cursor < fen.size()) {
        while (cursor < fen.size() && (fen[cursor] == ' ' || fen[cursor] == '\t')) ++cursor;
        if (cursor == fen.size()) break;
        const std::size_t start = cursor;
        while (cursor < fen.size() && fen[cursor] != ' ' && fen[cursor] != '\t') ++cursor;
        if (field_count == fields.size()) return fail("FEN must contain six fields");
        fields[field_count++] = std::string_view(fen).substr(start, cursor - start);
    }
    if (field_count != fields.size()) return fail("FEN must contain six fields");

    Features::Position candidate{};
    if (!parse_fen_board(fields[0], candidate)) return fail("invalid FEN board placement");
    bool white = false;
    if (fields[1] == "w") white = true;
    else if (fields[1] != "b") return fail("invalid FEN side to move");
    if (!valid_castling(fields[2])) return fail("invalid FEN castling field");
    if (fields[3] != "-") {
        if (fields[3].size() != 2 || fields[3][0] < 'a' || fields[3][0] > 'h' ||
            fields[3][1] != (white ? '6' : '3'))
            return fail("invalid FEN en-passant square");
    }
    std::int64_t halfmove = 0, fullmove = 0;
    if (!parse_integer(fields[4], halfmove) || halfmove < 0 ||
        !parse_integer(fields[5], fullmove) || fullmove < 1)
        return fail("invalid FEN move counters");
    if (!candidate.is_valid()) return fail("FEN must contain exactly one king per side and no overlapping pieces");
    for (std::size_t index = 0; index < candidate.num_pieces; ++index) {
        const Features::Piece& piece = candidate.pieces[index];
        if (piece.type == Features::PieceType::Pawn &&
            (Features::square_rank(piece.square) == 0 || Features::square_rank(piece.square) == 7))
            return fail("FEN contains a pawn on a promotion rank");
    }

    position = candidate;
    if (white_to_move) *white_to_move = white;
    if (error) error->clear();
    return true;
}

bool parse_dataset_line(const std::string& line, Example& example, std::string* error) {
    const std::size_t separator = line.find('\t');
    if (separator == std::string::npos) {
        if (error) *error = "dataset row must be FEN<TAB>eval_milli";
        return false;
    }
    std::string_view score(line.data() + separator + 1, line.size() - separator - 1);
    while (!score.empty() && (score.front() == ' ' || score.front() == '\t')) score.remove_prefix(1);
    while (!score.empty() && (score.back() == ' ' || score.back() == '\t' || score.back() == '\r')) score.remove_suffix(1);
    std::int64_t parsed_score = 0;
    if (!parse_integer(score, parsed_score) ||
        parsed_score < std::numeric_limits<std::int32_t>::min() ||
        parsed_score > std::numeric_limits<std::int32_t>::max()) {
        if (error) *error = "eval_milli must be a signed 32-bit integer";
        return false;
    }
    Example candidate;
    if (!parse_fen(line.substr(0, separator), candidate.position, nullptr, error)) return false;
    candidate.eval_milli = static_cast<std::int32_t>(parsed_score);
    example = candidate;
    if (error) error->clear();
    return true;
}

bool load_dataset(const std::string& path, std::vector<Example>& examples, std::string* error) {
    if (path.empty()) {
        if (error) *error = "empty training dataset path";
        return false;
    }
    std::ifstream input(path);
    if (!input) {
        if (error) *error = "cannot open training dataset";
        return false;
    }
    std::vector<Example> loaded;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        Example example;
        std::string row_error;
        if (!parse_dataset_line(line, example, &row_error)) {
            if (error) *error = "line " + std::to_string(line_number) + ": " + row_error;
            return false;
        }
        loaded.push_back(example);
    }
    if (input.bad()) {
        if (error) *error = "error while reading training dataset";
        return false;
    }
    if (loaded.empty()) {
        if (error) *error = "training dataset contains no examples";
        return false;
    }
    examples = std::move(loaded);
    if (error) error->clear();
    return true;
}

bool train(const std::vector<Example>& examples, const Options& options,
           Model::Model& output, Report* report, std::string* error) {
    if (examples.empty()) {
        if (error) *error = "training requires at least one example";
        return false;
    }
    if (options.epochs == 0 || !std::isfinite(options.learning_rate) || options.learning_rate <= 0.0f) {
        if (error) *error = "epochs and learning rate must be positive";
        return false;
    }
    if (options.validation_fraction < 0.0 || options.validation_fraction >= 1.0 ||
        !std::isfinite(options.validation_fraction)) {
        if (error) *error = "validation fraction must be in [0, 1)";
        return false;
    }
    for (const Example& example : examples) {
        if (!example.position.is_valid()) {
            if (error) *error = "training example contains an invalid position";
            return false;
        }
    }

    try {
        // Validation set: either a dedicated file (used verbatim) or a
        // deterministic fraction of the loaded examples, split with a seeded
        // RNG so the split is reproducible. Validation examples are never
        // passed to train_one().
        std::vector<Example> held_out;
        std::vector<bool> is_validation(examples.size(), false);
        if (!options.validation_path.empty()) {
            if (!load_dataset(options.validation_path, held_out, error)) return false;
        } else if (options.validation_fraction > 0.0) {
            const std::size_t validation_count =
                static_cast<std::size_t>(static_cast<double>(examples.size()) * options.validation_fraction);
            if (validation_count == 0 || validation_count >= examples.size()) {
                if (error) *error = "dataset is too small for the requested validation split; provide a separate validation file or set validation_fraction=0 explicitly";
                return false;
            }
            std::vector<std::size_t> order(examples.size());
            for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
            std::mt19937 split_random(options.seed ^ 0xA5A5A5A5u);
            std::shuffle(order.begin(), order.end(), split_random);
            held_out.reserve(validation_count);
            for (std::size_t index = 0; index < validation_count; ++index) {
                is_validation[order[index]] = true;
                held_out.push_back(examples[order[index]]);
            }
        }
        std::vector<Example> training;
        training.reserve(examples.size() - held_out.size());
        for (std::size_t index = 0; index < examples.size(); ++index)
            if (!is_validation[index]) training.push_back(examples[index]);
        if (training.empty()) {
            if (error) *error = "validation split consumed all examples; reduce validation_fraction or provide a separate validation file";
            return false;
        }
        const std::uint64_t current_dataset_signature = dataset_signature(training, held_out);

        std::mt19937 init_random(options.seed);
        FloatModel model = make_initial_model(init_random);
        CheckpointState checkpoint;
        bool resumed = false;
        if (!options.checkpoint_in.empty()) {
            if (!load_checkpoint(options.checkpoint_in, checkpoint, error)) return false;
            if (checkpoint.seed != options.seed ||
                checkpoint.learning_rate != options.learning_rate ||
                checkpoint.training_count != training.size() ||
                checkpoint.validation_count != held_out.size() ||
                checkpoint.dataset_signature != current_dataset_signature ||
                checkpoint.validation_fraction != options.validation_fraction ||
                checkpoint.had_validation != !held_out.empty()) {
                if (error) *error = "checkpoint does not match the current seed, learning rate, validation split or dataset";
                return false;
            }
            if (checkpoint.epochs_done > options.epochs) {
                if (error) *error = "requested total epochs is less than checkpoint progress";
                return false;
            }
            if (!finite_parameters(checkpoint.model) || !finite_parameters(checkpoint.best_model) ||
                !std::isfinite(checkpoint.best_validation_mae)) {
                if (error) *error = "checkpoint contains non-finite parameters";
                return false;
            }
            model = std::move(checkpoint.model);
            resumed = true;
        }
        Report local_report;
        local_report.training_examples = training.size();
        local_report.validation_examples = held_out.size();
        local_report.resumed = resumed;
        const ErrorMetrics initial_metrics = error_metrics(training, model);
        local_report.initial_mae_milli = initial_metrics.mean_absolute_error_milli;
        local_report.initial_max_error_milli = initial_metrics.max_absolute_error_milli;
        if (!std::isfinite(local_report.initial_mae_milli) ||
            !std::isfinite(local_report.initial_max_error_milli)) {
            if (error) *error = "non-finite initial training loss";
            return false;
        }

        FloatModel best_model = model;
        float best_validation_mae = std::numeric_limits<float>::infinity();
        std::size_t best_epoch = 0;
        bool use_validation = !held_out.empty();
        if (use_validation)
            best_validation_mae = error_metrics(held_out, model).mean_absolute_error_milli;
        if (resumed) {
            best_validation_mae = checkpoint.best_validation_mae;
            best_epoch = static_cast<std::size_t>(checkpoint.best_epoch);
            best_model = std::move(checkpoint.best_model);
        }
        const std::size_t start_epoch = resumed ? static_cast<std::size_t>(checkpoint.epochs_done) : 0;

        std::vector<std::size_t> order(training.size());
        for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
        for (std::size_t epoch = start_epoch; epoch < options.epochs; ++epoch) {
            // Per-epoch seeded permutation keeps resumed training bit-identical
            // to a continuous run without serializing RNG state (TODO #13).
            for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
            std::mt19937 epoch_random(options.seed ^
                (0x9E3779B9u * static_cast<std::uint32_t>(epoch + 1)));
            std::shuffle(order.begin(), order.end(), epoch_random);
            for (std::size_t index : order)
                train_one(training[index], model, options.learning_rate);
            if (!finite_parameters(model)) {
                if (error) *error = "training produced non-finite parameters in epoch " +
                    std::to_string(epoch + 1) + "; aborting (lower the learning rate)";
                return false;
            }
            ++local_report.epochs_run;
            Report::EpochMetrics metrics;
            metrics.epoch = epoch + 1;
            const ErrorMetrics train_metrics = error_metrics(training, model);
            metrics.train_mae_milli = train_metrics.mean_absolute_error_milli;
            metrics.train_max_error_milli = train_metrics.max_absolute_error_milli;
            if (!std::isfinite(metrics.train_mae_milli) ||
                !std::isfinite(metrics.train_max_error_milli)) {
                if (error) *error = "non-finite training loss in epoch " + std::to_string(epoch + 1);
                return false;
            }
            if (use_validation) {
                const ErrorMetrics validation_metrics = error_metrics(held_out, model);
                metrics.validation_mae_milli = validation_metrics.mean_absolute_error_milli;
                metrics.validation_max_error_milli = validation_metrics.max_absolute_error_milli;
                if (!std::isfinite(metrics.validation_mae_milli) ||
                    !std::isfinite(metrics.validation_max_error_milli)) {
                    if (error) *error = "non-finite validation loss in epoch " + std::to_string(epoch + 1);
                    return false;
                }
                if (metrics.validation_mae_milli < best_validation_mae) {
                    best_validation_mae = metrics.validation_mae_milli;
                    best_model = model;
                    best_epoch = epoch + 1;
                }
            } else {
                metrics.validation_mae_milli = std::numeric_limits<float>::quiet_NaN();
                metrics.validation_max_error_milli = std::numeric_limits<float>::quiet_NaN();
                best_model = model;
                best_epoch = epoch + 1;
            }
            local_report.epoch_history.push_back(metrics);
        }

        if (!options.checkpoint_out.empty()) {
            CheckpointState saved;
            saved.seed = options.seed;
            saved.learning_rate = options.learning_rate;
            saved.epochs_requested = options.epochs;
            saved.validation_fraction = options.validation_fraction;
            saved.training_count = training.size();
            saved.validation_count = held_out.size();
            saved.dataset_signature = current_dataset_signature;
            saved.epochs_done = start_epoch + local_report.epochs_run;
            saved.best_validation_mae = use_validation ? best_validation_mae : 0.0f;
            saved.had_validation = use_validation;
            saved.best_epoch = best_epoch;
            saved.model = model;
            saved.best_model = best_model;
            if (!save_checkpoint(options.checkpoint_out, saved, error)) return false;
        }

        const ErrorMetrics final_train_metrics = error_metrics(training, best_model);
        local_report.final_mae_milli = final_train_metrics.mean_absolute_error_milli;
        local_report.final_max_error_milli = final_train_metrics.max_absolute_error_milli;
        local_report.final_validation_mae_milli = use_validation ? best_validation_mae : 0.0f;
        if (use_validation)
            local_report.final_validation_max_error_milli =
                error_metrics(held_out, best_model).max_absolute_error_milli;
        local_report.best_epoch = best_epoch;
        if (!std::isfinite(local_report.final_mae_milli) ||
            !std::isfinite(local_report.final_max_error_milli) ||
            !std::isfinite(local_report.final_validation_max_error_milli) ||
            !finite_parameters(best_model)) {
            if (error) *error = "training produced non-finite parameters or loss";
            return false;
        }
        local_report.saturated_parameters = export_model(best_model, output);
        NNUE evaluator;
        if (!evaluator.load_model(output, error)) {
            if (error) *error = "cannot validate exported model: " + *error;
            return false;
        }
        std::size_t parity_examples = 0;
        float max_export_error = 0.0f;
        auto check_parity = [&](const std::vector<Example>& set) {
            for (const Example& example : set) {
                const float expected = forward(example, best_model).prediction * ScoreScale;
                const float actual = static_cast<float>(evaluator.evaluate(example.position));
                max_export_error = std::max(max_export_error, std::abs(actual - expected));
                if (++parity_examples >= 64) return;
            }
        };
        check_parity(training);
        if (parity_examples < 64) check_parity(held_out);
        local_report.max_export_prediction_error_milli = max_export_error;
        if (max_export_error > 1.0f) {
            if (error) *error = "quantized training and exported NNUE predictions differ by " +
                std::to_string(max_export_error) + " mP (tolerance: 1 mP)";
            return false;
        }
        if (report) *report = std::move(local_report);
        if (error) error->clear();
        return true;
    } catch (const std::bad_alloc&) {
        if (error) *error = "out of memory while training NNUE";
        return false;
    }
}

} // namespace TAPA::Training
