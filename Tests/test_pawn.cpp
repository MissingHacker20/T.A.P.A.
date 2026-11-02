#include <iostream>
#include "Source Code/Features/features.h"

int main() {
    using namespace TAPA::Features;
    
    // Create a simple position: white pawn on e2 (square index? Let's assume a1=0, b1=1, ... h8=63)
    // e2: file e (4), rank 2 -> square = (rank-1)*8 + file = (1)*8 + 4 = 12? Actually rank1 index0, rank2 index1 => square = 1*8 + 4 = 12.
    Position pos;
    pos.num_pieces = 1;
    pos.pieces[0] = {PieceType::Pawn, PieceColor::White, 12}; // e2
    
    // Add a black pawn on e7 (rank 7 -> index 6, file 4 => square = 6*8 + 4 = 52)
    pos.pieces[1] = {PieceType::Pawn, PieceColor::Black, 52};
    pos.num_pieces = 2;
    
    // Add a white knight on g1 (rank1 index0, file6 => square = 0*8+6=6)
    pos.pieces[2] = {PieceType::Knight, PieceColor::White, 6};
    pos.num_pieces = 3;
    
    // Add black king on e8 (rank7 index6, file4 => square = 6*8+4=52? Wait that's e7; e8 is rank8 index7, file4 => 7*8+4=60)
    pos.pieces[3] = {PieceType::King, PieceColor::Black, 60};
    pos.num_pieces = 4;
    
    // White king on e1 (rank0 index0, file4 => 4)
    pos.pieces[4] = {PieceType::King, PieceColor::White, 4};
    pos.num_pieces = 5;
    
    // Compute pawn structure key
    PawnStructureKey key = compute_pawn_structure_key(pos);
    std::cout << "PawnStructureKey: 0x" << std::hex << key << std::dec << std::endl;
    
    // Generate features
    FeatureSet fs;
    generate_features(pos, fs);
    std::cout << "Number of features (excluding structural pawns): " << fs.size << std::endl;
    for (size_t i = 0; i < fs.size; ++i) {
        std::cout << "Feature[" << i << "] index=" << fs.features[i].index << ", value=" << fs.features[i].value << std::endl;
    }
    
    return 0;
}