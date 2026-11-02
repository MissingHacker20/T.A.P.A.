// test_main.cpp
#include <iostream>
#include "Source Code/Features/features.h"
#include "Source Code/Accumulator/accumulator.h"
#include "Source Code/Network/network.h"
#include "Source Code/NNUE/nnue.h"

int main() {
    using namespace TAPA;

    NNUE nnue;
    nnue.initialize(0);

    Features::Position pos;
    pos.num_pieces = 0;
    pos.pieces[pos.num_pieces++] = {Features::PieceType::King, Features::PieceColor::White, 4};
    pos.pieces[pos.num_pieces++] = {Features::PieceType::King, Features::PieceColor::Black, 5};

    int32_t score = nnue.evaluate(pos);
    std::cout << "Score: " << score << std::endl;

    // Test accumulator
    TAPA::Accumulator::Accumulator acc;
    acc.initialize(5);
    Features::Feature f{100, 1};
    const int32_t* w = get_weights_for_feature(f.index);
    acc.add_feature(f, w);
    // Check first neuron
    std::cout << "Accumulator[0] after add: " << acc.get_data()[0] << " (expected " << 5 + w[0] << ")" << std::endl;

    return 0;
}