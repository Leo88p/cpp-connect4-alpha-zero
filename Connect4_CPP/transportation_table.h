#pragma once
#include <cstdint>
#include <vector>
#include <cstring> // Required for std::memset

namespace Connect4 {
    enum class EntryType : uint8_t {
        EXACT,
        LOWER_BOUND,
        UPPER_BOUND
    };

    struct TTEntry {
        uint64_t hash_key = 0;
        int8_t score = 0;
        uint8_t depth = 0;
        EntryType type = EntryType::EXACT;
        int8_t best_move = -1; // Stores the best column for move ordering
    };

    class TranspositionTable {
    private:
        std::vector<TTEntry> table;
        size_t mask;

    public:
        explicit TranspositionTable(size_t num_elements) {
            size_t size = 1;
            while (size < num_elements) size <<= 1;
            table.resize(size);
            mask = size - 1;
        }

        void clear() {
            std::memset(table.data(), 0, table.size() * sizeof(TTEntry));
        }

        bool lookup(uint64_t hash_key, TTEntry& out_entry) const {

            const TTEntry& entry = table[hash_key & mask];
            if (entry.hash_key == hash_key) {
                out_entry = entry;
                return true;
            }
            return false;
        }

        void store(uint64_t hash_key, int8_t score, uint8_t depth, EntryType type, int8_t best_move) {
            TTEntry& entry = table[hash_key & mask];

            if (entry.hash_key == 0 || entry.hash_key == hash_key || depth >= entry.depth) {
                entry.hash_key = hash_key;
                entry.score = score;
                entry.depth = depth;
                entry.type = type;
                entry.best_move = best_move;
            }
        }
    };
}