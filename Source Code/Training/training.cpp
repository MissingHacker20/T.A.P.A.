#include "training.h"
#include "../Network/network.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
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
    float pawn_bias = 0.0f;
    std::array<Vector256, 64> pawn_weights{};
    float piece_bias = 0.0f;
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

float quantize(float value, float scale) noexcept {
    return std::round(value * scale) / scale;
}

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
        pawn[neuron] = quantize(model.pawn_bias, PawnEmbeddingScale);

    const Features::PawnStructureKey key = Features::make_pawn_structure_key(example.position);
    for (std::size_t bit = 0; bit < 64; ++bit) {
        if ((key & (Features::PawnStructureKey{1} << bit)) == 0) continue;
        for (std::size_t neuron = 0; neuron < pawn.size(); ++neuron)
            pawn[neuron] += quantize(model.pawn_weights[bit][neuron], PawnEmbeddingScale);
    }

    Features::FeatureSet features;
    Features::generate_features(example.position, features);
    for (std::size_t feature_index = 0; feature_index < features.size; ++feature_index) {
        const Features::Feature& feature = features.features[feature_index];
        const Vector256& row = model.piece_weights[feature.index];
        Vector256& destination = feature.perspective == Features::PieceColor::White ? piece_white : piece_black;
        for (std::size_t neuron = 0; neuron < destination.size(); ++neuron)
            destination[neuron] += quantize(row[neuron], PieceEmbeddingScale);
    }

    for (std::size_t neuron = 0; neuron < pawn.size(); ++neuron) {
        const float common = pawn[neuron] + quantize(model.piece_bias, PieceEmbeddingScale);
        input[neuron] = common + piece_white[neuron];
        input[neuron + Accumulator::NeuronCount] = common + piece_black[neuron];
    }
}

template<int In, int Out>
float quantized_weight(const FloatLayer<In, Out>& layer, int input, int output,
                       float scale) noexcept {
    return quantize(layer.weights[static_cast<std::size_t>(input)][static_cast<std::size_t>(output)], scale);
}

ForwardState forward(const Example& example, const FloatModel& model) noexcept {
    ForwardState state;
    build_inputs(example, model, state.input);

    constexpr float hidden1_limit = static_cast<float>(ActivationLimit) / Hidden1Scale;
    constexpr float hidden2_limit = static_cast<float>(ActivationLimit) / Hidden2Scale;
    for (int output = 0; output < Network::HiddenSize1; ++output) {
        float sum = quantize(model.layer1.bias[static_cast<std::size_t>(output)], Hidden1Scale);
        for (int input = 0; input < Network::InputSize; ++input)
            sum += state.input[static_cast<std::size_t>(input)] *
                quantized_weight(model.layer1, input, output, Hidden1Scale / PieceEmbeddingScale);
        state.hidden1[static_cast<std::size_t>(output)] = clipped_relu(sum, hidden1_limit);
    }

    for (int output = 0; output < Network::HiddenSize2; ++output) {
        float sum = quantize(model.layer2.bias[static_cast<std::size_t>(output)], Hidden2Scale);
        for (int input = 0; input < Network::HiddenSize1; ++input)
            sum += state.hidden1[static_cast<std::size_t>(input)] *
                quantized_weight(model.layer2, input, output, Hidden2Scale / Hidden1Scale);
        state.hidden2[static_cast<std::size_t>(output)] = clipped_relu(sum, hidden2_limit);
    }

    state.prediction = quantize(model.output.bias[0], ScoreScale);
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
    for (int input = 0; input < Network::InputSize; ++input) {
        const std::size_t index = static_cast<std::size_t>(input);
        float gradient = 0.0f;
        for (int output = 0; output < Network::HiddenSize1; ++output) {
            const std::size_t output_index = static_cast<std::size_t>(output);
            gradient += hidden1_preactivation_gradient[output_index] *
                quantized_weight(model.layer1, input, output, Hidden1Scale / PieceEmbeddingScale);
        }
        input_gradient[index] = gradient;
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
    model.piece_bias -= learning_rate * piece_bias_gradient;
    model.pawn_bias -= learning_rate * pawn_bias_gradient;

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

float mean_absolute_error(const std::vector<Example>& examples, const FloatModel& model) noexcept {
    double absolute_error = 0.0;
    for (const Example& example : examples) {
        const float target = static_cast<float>(example.eval_milli) / ScoreScale;
        absolute_error += std::abs(forward(example, model).prediction - target);
    }
    return static_cast<float>(absolute_error * (ScoreScale / static_cast<double>(examples.size())));
}

bool finite_parameters(const FloatModel& model) noexcept {
    auto finite_range = [](const auto& range) {
        for (const auto& value : range)
            if (!std::isfinite(value)) return false;
        return true;
    };
    if (!std::isfinite(model.pawn_bias) || !std::isfinite(model.piece_bias)) return false;
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

template<int In, int Out>
void export_layer(const FloatLayer<In, Out>& source,
                  Network::Layer<In, Out>& destination, float weight_scale, float bias_scale) noexcept {
    for (int input = 0; input < In; ++input)
        for (int output = 0; output < Out; ++output)
            destination.weights[static_cast<std::size_t>(input)][static_cast<std::size_t>(output)] =
                to_i32(source.weights[static_cast<std::size_t>(input)][static_cast<std::size_t>(output)] * weight_scale);
    for (int output = 0; output < Out; ++output)
        destination.bias[static_cast<std::size_t>(output)] =
            to_i32(source.bias[static_cast<std::size_t>(output)] * bias_scale);
}

void export_model(const FloatModel& source, Model::Model& destination) noexcept {
    destination.pawn_weights.bias = to_i32(source.pawn_bias * PawnEmbeddingScale);
    destination.piece_weights.bias = to_i32(source.piece_bias * PieceEmbeddingScale);
    for (std::size_t neuron = 0; neuron < Accumulator::NeuronCount; ++neuron) {
        for (std::size_t bit = 0; bit < 64; ++bit)
            destination.pawn_weights.bit_weights[bit][neuron] =
                to_i16(source.pawn_weights[bit][neuron] * PawnEmbeddingScale);
        for (std::size_t feature = 0; feature < Accumulator::PieceFeatureCount; ++feature)
            destination.piece_weights.feature_weights[feature][neuron] =
                to_i16(source.piece_weights[feature][neuron] * PieceEmbeddingScale);
    }
    export_layer(source.layer1, destination.network_weights.layer1,
                 Hidden1Scale / PieceEmbeddingScale, Hidden1Scale);
    export_layer(source.layer2, destination.network_weights.layer2,
                 Hidden2Scale / Hidden1Scale, Hidden2Scale);
    export_layer(source.output, destination.network_weights.output,
                 ScoreScale / Hidden2Scale, ScoreScale);
    destination.network_weights.activation_limit = ActivationLimit;
    destination.network_weights.output_scale = 1;
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
    for (const Example& example : examples) {
        if (!example.position.is_valid()) {
            if (error) *error = "training example contains an invalid position";
            return false;
        }
    }

    try {
        std::mt19937 random(options.seed);
        FloatModel model = make_initial_model(random);
        Report local_report;
        local_report.initial_mae_milli = mean_absolute_error(examples, model);
        if (!std::isfinite(local_report.initial_mae_milli)) {
            if (error) *error = "non-finite initial training loss";
            return false;
        }
        std::vector<std::size_t> order(examples.size());
        for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
        for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), random);
            for (std::size_t index : order)
                train_one(examples[index], model, options.learning_rate);
        }
        local_report.final_mae_milli = mean_absolute_error(examples, model);
        if (!std::isfinite(local_report.final_mae_milli) || !finite_parameters(model)) {
            if (error) *error = "training produced non-finite parameters or loss";
            return false;
        }
        export_model(model, output);
        if (report) *report = local_report;
        if (error) error->clear();
        return true;
    } catch (const std::bad_alloc&) {
        if (error) *error = "out of memory while training NNUE";
        return false;
    }
}

} // namespace TAPA::Training
