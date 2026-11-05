#include "pawn_accumulator.h"
#include <limits>

namespace TAPA::Accumulator
{

    void PawnAccumulator::reset(PawnAccumulatorValue bias) noexcept
    {
        exact_data_.fill(bias);
        data_.fill(bias);
    }

    namespace {
        PawnAccumulatorValue saturate_accumulator_value(std::int64_t value) noexcept
        {
            return static_cast<PawnAccumulatorValue>(value < std::numeric_limits<PawnAccumulatorValue>::min()
                ? std::numeric_limits<PawnAccumulatorValue>::min()
                : value > std::numeric_limits<PawnAccumulatorValue>::max()
                    ? std::numeric_limits<PawnAccumulatorValue>::max() : value);
        }
    }

    void PawnAccumulator::reconstruct(Features::PawnStructureKey key, const PawnWeights &weights) noexcept
    {
        reset(weights.bias);
        for (std::size_t bit = 0; bit < 64; ++bit)
        {
            if (((key >> bit) & 1ULL) == 0)
                continue;
            for (std::size_t neuron = 0; neuron < PawnNeuronCount; ++neuron)
                exact_data_[neuron] += weights.bit_weights[bit][neuron];
        }
        for (std::size_t neuron = 0; neuron < PawnNeuronCount; ++neuron)
            data_[neuron] = saturate_accumulator_value(exact_data_[neuron]);
    }

    void PawnAccumulator::update(Features::PawnStructureKey old_key,
                                 Features::PawnStructureKey new_key,
                                 const PawnWeights &weights) noexcept
    {
        const Features::PawnStructureKey removed = old_key & ~new_key;
        const Features::PawnStructureKey added = new_key & ~old_key;
        for (std::size_t bit = 0; bit < 64; ++bit)
        {
            const Features::PawnStructureKey mask = Features::PawnStructureKey{1} << bit;
            if ((removed & mask) != 0)
                for (std::size_t neuron = 0; neuron < PawnNeuronCount; ++neuron)
                    exact_data_[neuron] -= weights.bit_weights[bit][neuron];
            if ((added & mask) != 0)
                for (std::size_t neuron = 0; neuron < PawnNeuronCount; ++neuron)
                    exact_data_[neuron] += weights.bit_weights[bit][neuron];
        }
        for (std::size_t neuron = 0; neuron < PawnNeuronCount; ++neuron)
            data_[neuron] = saturate_accumulator_value(exact_data_[neuron]);
    }

    const PawnAccumulator *PawnAccumulatorCache::find(Features::PawnStructureKey key) const noexcept
    {
        for (const Entry &entry : entries_)
            if (entry.valid && entry.key == key) return &entry.accumulator;
        return nullptr;
    }

    const PawnAccumulator *PawnAccumulatorCache::touch(Features::PawnStructureKey key) noexcept
    {
        ++clock_;
        for (Entry &entry : entries_)
            if (entry.valid && entry.key == key) {
                entry.age = clock_;
                ++stats_.hits;
                return &entry.accumulator;
            }
        ++stats_.misses; // touch() misses must be reflected in the stats too
        return nullptr;
    }

    const PawnAccumulator &PawnAccumulatorCache::get(Features::PawnStructureKey key) noexcept
    {
        ++clock_;
        for (Entry &entry : entries_)
        {
            if (entry.valid && entry.key == key)
            {
                entry.age = clock_;
                ++stats_.hits;
                return entry.accumulator;
            }
        }
        Entry *replacement = &entries_[0];
        for (Entry &entry : entries_)
        {
            if (!entry.valid)
            {
                replacement = &entry;
                break;
            }
            if (entry.age < replacement->age)
                replacement = &entry;
        }
        replacement->key = key;
        replacement->accumulator.reconstruct(key, *weights_);
        replacement->age = clock_;
        replacement->valid = true;
        ++stats_.misses;
        ++insertions_; // get() creates an entry on a miss
        return replacement->accumulator;
    }

    void PawnAccumulatorCache::insert(Features::PawnStructureKey key, const PawnAccumulator &accumulator) noexcept
    {
        ++clock_;
        Entry *replacement = nullptr;
        for (Entry &entry : entries_) {
            if (entry.valid && entry.key == key) {
                replacement = &entry;
                break;
            }
            if (!entry.valid) {
                replacement = &entry;
                break;
            }
            if (replacement == nullptr || entry.age < replacement->age)
                replacement = &entry;
        }
        replacement->key = key;
        replacement->accumulator = accumulator;
        replacement->age = clock_;
        replacement->valid = true;
        ++insertions_;
    }

    bool PawnAccumulatorCache::contains(Features::PawnStructureKey key) const noexcept
    {
        for (const Entry &entry : entries_)
            if (entry.valid && entry.key == key)
                return true;
        return false;
    }

    void PawnAccumulatorCache::clear() noexcept
    {
        for (Entry &entry : entries_)
            entry.valid = false;
        clock_ = 0;
        stats_ = {};
        insertions_ = 0;
    }

} // namespace TAPA::Accumulator
