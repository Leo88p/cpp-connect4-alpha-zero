#include <cstdint>
#include <array>

namespace Connect4 {
    class ConstexprRandom {
    private:
        uint64_t state;
    public:
        constexpr explicit ConstexprRandom(uint64_t seed) : state(seed == 0 ? 5489ULL : seed) {}

        constexpr uint64_t next() {
            state ^= (state << 13);
            state ^= (state >> 7);
            state ^= (state << 17);
            return state;
        }
    };

    class ZobristTable {
    public:
        std::array<std::array<uint64_t, 2>, 49> table{};
        uint64_t black_turn_key = 0;

        constexpr ZobristTable() {
            ConstexprRandom rng(0x123456789ABCDEFULL);

            for (int bit = 0; bit < 49; ++bit) {
                table[bit][0] = rng.next();
                table[bit][1] = rng.next();
            }
            black_turn_key = rng.next();
        }
    };

    inline constexpr ZobristTable ZOBRIST;
}