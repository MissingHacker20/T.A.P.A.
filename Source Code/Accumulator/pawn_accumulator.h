#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include "../Features/features.h"

namespace TAPA::Accumulator
{

    constexpr std::size_t PawnNeuronCount = 256;
    using PawnAccumulatorValue = std::int32_t;
    using PawnWeight = std::int16_t;
    constexpr std::size_t MaxPawnContributions = 64;

    struct PawnWeights
    {
        PawnAccumulatorValue bias = 0;
        std::array<std::array<PawnWeight, PawnNeuronCount>, 64> bit_weights{};
    };

    class PawnAccumulator
    {
    public:
        void reset(PawnAccumulatorValue bias = 0) noexcept;
        void reconstruct(Features::PawnStructureKey key, const PawnWeights &weights) noexcept;
        void update(Features::PawnStructureKey old_key, Features::PawnStructureKey new_key,
                    const PawnWeights &weights) noexcept;
        const std::array<PawnAccumulatorValue, PawnNeuronCount> &data() const noexcept { return data_; }
        friend bool operator==(const PawnAccumulator &lhs, const PawnAccumulator &rhs) noexcept
        {
            return lhs.data_ == rhs.data_;
        }

    private:
        std::array<PawnAccumulatorValue, PawnNeuronCount> data_{};
    };

    struct PawnCacheStats
    {
        std::size_t hits = 0;
        std::size_t misses = 0;
    };

    class PawnAccumulatorCache
    {
    public:
        static constexpr std::size_t Capacity = 64;
        explicit PawnAccumulatorCache(const PawnWeights &weights) noexcept : weights_(&weights) {}
        void set_weights(const PawnWeights &weights) noexcept
        {
            weights_ = &weights;
            clear();
        }
        const PawnAccumulator &get(Features::PawnStructureKey key) noexcept;
        void insert(Features::PawnStructureKey key, const PawnAccumulator &accumulator) noexcept;
        const PawnAccumulator *find(Features::PawnStructureKey key) const noexcept;
        const PawnAccumulator *touch(Features::PawnStructureKey key) noexcept;
        std::size_t insertions() const noexcept { return insertions_; }
        void clear() noexcept;
        PawnCacheStats stats() const noexcept { return stats_; }
        static constexpr std::size_t capacity() noexcept { return Capacity; }
        bool contains(Features::PawnStructureKey key) const noexcept;

    private:
        struct Entry
        {
            Features::PawnStructureKey key = 0;
            PawnAccumulator accumulator{};
            std::uint64_t age = 0;
            bool valid = false;
        };
        const PawnWeights *weights_;
        std::array<Entry, Capacity> entries_{};
        std::uint64_t clock_ = 0;
        PawnCacheStats stats_{};
        std::size_t insertions_ = 0;
    };

} // namespace TAPA::Accumulator
