// main.cpp - Simple test driver for T.A.P.A. NNUE
#include <iostream>
#include "Source Code/Features/features.h"
#include "Source Code/Accumulator/accumulator.h"
#include "Source Code/Network/network.h"
#include "Source Code/NNUE/nnue.h"

int main() {
    std::cout << "=== T.A.P.A. NNUE Test ===" << std::endl;
    
    // Initialize NNUE
    TAPA::NNUE nnue;
    nnue.initialize(0); // zero bias
    
    // Create test position: white king e4 (4), black king e5 (5)
    TAPA::Features::Position pos;
    pos.num_pieces = 0;
    pos.pieces[pos.num_pieces++] = {TAPA::Features::PieceType::King, TAPA::Features::PieceColor::White, 4};
    pos.pieces[pos.num_pieces++] = {TAPA::Features::PieceType::King, TAPA::Features::PieceColor::Black, 5};
    
    // Evaluate position
    int32_t score = nnue.evaluate(pos);
    std::cout << "Position: White Ke4, Black Ke5" << std::endl;
    std::cout << "Evaluation: " << score << std::endl;
    
    // Test feature generation
    TAPA::Features::FeatureSet fs;
    TAPA::Features::generate_features(pos, fs);
    std::cout << "Number of features: " << fs.size << std::endl;
    
    // Test accumulator
    TAPA::Accumulator::Accumulator acc;
    acc.initialize(10); // bias 10
    TAPA::Features::Feature f{100, 1}; // feature index 100, value 1
    const int32_t* weights = TAPA::get_weights_for_feature(f.index);
    acc.add_feature(f, weights);
    std::cout << "Accumulator[0] after adding feature: " << acc.get_data()[0] 
              << " (expected: " << 10 + weights[0] << ")" << std::endl;
    
    // Test move: white king e4 -> d4 (3)
    TAPA::Features::Position pos2 = pos;
    for (size_t i = 0; i < pos2.num_pieces; ++i) {
        if (pos2.pieces[i].color == TAPA::Features::PieceColor::White &&
            pos2.pieces[i].type == TAPA::Features::PieceType::King) {
            pos2.pieces[i].square = 3;
            break;
        }
    }
    
    nnue.update(pos, pos2);
    int32_t score2 = nnue.evaluate(pos2);
    std::cout << "After moving white king to d4:" << std::endl;
    std::cout << "Evaluation: " << score2 << std::endl;
    
    // Test push/pop
    nnue.push();
    
    // Another move: black king e5 -> d5
    TAPA::Features::Position pos3 = pos2;
    for (size_t i = 0; i < pos3.num_pieces; ++i) {
        if (pos3.pieces[i].color == TAPA::Features::PieceColor::Black &&
            pos3.pieces[i].type == TAPA::Features::PieceType::King) {
            pos3.pieces[i].square = 3;
            break;
        }
    }
    
    nnue.update(pos2, pos3);
    int32_t score3 = nnue.evaluate(pos3);
    std::cout << "After moving black king to d5:" << std::endl;
    std::cout << "Evaluation: " << score3 << std::endl;
    
    nnue.pop();
    int32_t score_after_pop = nnue.evaluate(pos2);
    std::cout << "After pop (back to position after first move):" << std::endl;
    std::cout << "Evaluation: " << score_after_pop << std::endl;
    
    // Verify push/pop works
    if (score2 == score_after_pop) {
        std::cout << "✓ Push/pop test PASSED" << std::endl;
    } else {
        std::cout << "✗ Push/pop test FAILED" << std::endl;
    }
    
    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}