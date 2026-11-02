# T.A.P.A.

T.A.P.A. (Technologia Aestimationis Pignorium Absentium) is an NNUE evaluation library designed for integration with the MissingPawn chess engine.

## Relationship to MissingPawn

T.A.P.A. is the evaluation layer only. MissingPawn owns board representation, move generation and search; it feeds T.A.P.A. a `Features::Position` adapter plus a `Features::Move`, and receives a centipawn score. Accumulators and the pawn-structure cache live per NNUE instance (per search thread); no global mutable state is shared.

## Architecture
```text
Position → 64-bit PawnStructureKey → Features → Accumulators → Network → Evaluation
```

`PawnStructureKey` is exactly a canonical `uint64_t`: White uses bits `0..31`, with file `a..h` in increasing order inside each rank byte. White's rank-2..5 bytes occupy bits `0..7`, `8..15`, `16..23`, and `24..31`; Black's rank-7,6,5,4 bytes occupy bits `32..39`, `40..47`, `48..55`, and `56..63`. The mapping is independent of host endianness. Advanced pawns (White ranks 6–7 and Black ranks 3–2) are emitted as ordinary color/square/type `Pawn` features and omitted from the key. Feature indices are canonical HalfKP-like `(king square, piece color, type, piece square)` slots, with a separate perspective tag.

### Accumulators and cache

Evaluation sums three embedding blocks per network input:

* `PawnAccumulator` — 256 × `int32_t`, seeded with an `int32_t` bias; each set bit of the 64-bit `PawnStructureKey` adds one `[64][256]` row of `int16_t` weights. Cached per NNUE instance in `PawnAccumulatorCache`: 64 entries, exact-key lookup (no hash buckets), LRU replacement.
* `PieceKingAccumulator` — one per perspective, 256 × `int32_t`, `int16_t` weights indexed by canonical feature index.
* Feature-relative base layer (`NNUE::weights_`) — a per-feature `int32_t` row added into both perspectives' network inputs.

On a move, the pawn accumulator is fetched from cache when `PawnStructureKey` changes (or reused when it does not), and piece features are applied incrementally from the move `FeatureDelta`. A full reconstruction (`evaluate`) is the reference path; differential tests assert `incremental == reconstruction` after every ply.

### Model file

`TAPA::Model::load()` validates: magic `TAPANNUE`, `FormatVersion`, an endianness marker (`0x01020304`, little-endian payload), embedding width, payload size, an FNV-1a checksum, architecture sizes (512→32→32→1) and two format-version tags (feature-index schema, pawn-structure encoding). Weights are stored little-endian, biases after weight tables, pawn key bit rows first. Models that fail any check are rejected with a descriptive error; incompatible architectures and corrupted payloads are covered by tests.

The fixed-size `Position` adapter avoids position-copy allocations in evaluation and can be populated from MissingPawn's native position API. `Move` is a move descriptor, not a complete legal chess move generator: the host engine remains the authority for check legality, rights, and game state. `apply_move` validates piece geometry and handles captures, en-passant, promotion, and castling for adapter/test use. `update_features(before, move, after)` creates a move-oriented delta, rebuilding features only for a perspective whose king square changed. A separate 256-neuron Pawn Accumulator (`int32_t` values, `int16_t` weights) is cached in a per-NNUE, bounded 64-entry LRU using the full `uint64_t` key. The model format is little-endian and versioned (v3), with `TAPANNUE` magic, endianness marker, architecture and feature/key schema versions, ordered signed weights/biases, and FNV-1a payload checksum. Network/model weights are immutable and owned per `NNUE` after loading; each NNUE's mutable state and cache are worker-local. `NNUE::evaluate()` reconstructs owned state; after that `update()` and `evaluate_current()` use the owned incremental state.

## Build and tests
Requirements: C++20 and CMake 3.20 or newer.

```text
cmake -S . -B build -DTAPA_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

Enable the opt-in microbenchmark target with `-DTAPA_BUILD_BENCHMARKS=ON` and run `build/tapa_benchmarks` (or the corresponding configuration-specific executable path on multi-config generators). The benchmark reports elapsed microseconds and structure/weight sizes; run optimized Release builds and compare measurements on the same machine. Tests include deterministic move sequences and differential reconstruction checks.

Tests and benchmark sources are in `Tests/`. No training-data directory is part of the repository; generated build outputs and compiled artifacts are ignored.
