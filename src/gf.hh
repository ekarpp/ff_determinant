/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#ifndef GF_H
#define GF_H

#include <stdint.h>
#include <bitset>
#include <iostream>
#include <immintrin.h>

#include "global.hh"

namespace ff_util
{
    const uint64_t gf_n = 16;
    /* x^16 + x^5 + x^3 + x^2 +  1 */
    const uint64_t gf_mod = 0x1002D;
    const uint64_t gf_mask = 0xFFFF;

    /* returns q s.t. for some r,
     * a = q*b + r is the division relation
     */
    inline uint64_t quo(uint64_t a, uint64_t b)
    {
        uint64_t q = 0b0;
        int degb = util::log2(b);
        while (a >= b)
        {
            int shift = util::log2(a);
            shift -= degb;
            /* shift = deg(a) - deg(b) */
            q ^= (1ll << shift);
            a ^= (b << shift);
        }
        return q;
    }

    /* returns r s.t. for some q,
     * a = q*field.mod + r is the division relation (in Z(2^n))
     */
    inline uint64_t rem(uint64_t a)
    {
        uint64_t lo = a & ff_util::gf_mask;
        uint64_t hi = a >> ff_util::gf_n;

        uint64_t r = hi ^ (hi >> 14) ^ (hi >> 13) ^ (hi >> 11);
        r ^= (r << 2) ^ (r << 3) ^ (r << 5);
        r &= ff_util::gf_mask;
        return r ^ lo;
    }

#ifdef AVX512
    inline __m512i wide_mul(__m512i a, __m512i b)
    {
        const __m512i mask = _mm512_set1_epi64(ff_util::gf_mask);
        /* 16 hi bits of each multiplication */
        __m512i hi = _mm512_setzero_si512();
        /* 16 lo bits of each multiplication */
        __m512i lo = _mm512_setzero_si512();
        #pragma GCC unroll 32
        for (int i = 0; i < 4; i++)
        {
            const __m512i aa = _mm512_and_si512(
                _mm512_srli_epi64(a, 16*(3-i)),
                mask
            );
            const __m512i bb = _mm512_and_si512(
                _mm512_srli_epi64(b, 16*(3-i)),
                mask
            );

            const __m512i prod = _mm512_or_si512(
                _mm512_clmulepi64_epi128(
                    aa,
                    bb,
                    0x00
                ),
                _mm512_shuffle_epi32(
                    _mm512_clmulepi64_epi128(
                        aa,
                        bb,
                        0x11
                    ),
                    _MM_PERM_BADC
                )
            );
            hi = _mm512_or_si512(
                _mm512_slli_epi64(hi, 16),
                _mm512_srli_epi64(prod, 16)
            );
            lo = _mm512_or_si512(
                _mm512_slli_epi64(lo, 16),
                _mm512_and_si512(prod, mask)
            );
        }

        const __m512i tmp = _mm512_xor_si512(
            hi,
            _mm512_xor_si512(
                _mm512_srli_epi16(hi, 14),
                _mm512_xor_si512(
                    _mm512_srli_epi16(hi, 13),
                    _mm512_srli_epi16(hi, 11)
                )
            )
        );

        const __m512i rem = _mm512_xor_si512(
            tmp,
            _mm512_xor_si512(
                _mm512_slli_epi16(tmp, 2),
                _mm512_xor_si512(
                    _mm512_slli_epi16(tmp, 3),
                    _mm512_slli_epi16(tmp, 5)
                )
            )
        );

        return _mm512_xor_si512(rem, lo);
    }
#else
    inline __m256i wide_mul(__m256i a, __m256i b)
    {
        /* al/bl might not be needed, just use a/b */
        const __m256i mask = _mm256_set1_epi32(ff_util::gf_mask);


        /* 16 hi bits of each multiplication */
        __m256i hi = _mm256_setzero_si256();
        /* 16 lo bits of each multiplication */
        __m256i lo = _mm256_setzero_si256();
        #pragma GCC unroll 32
        for (int i = 0; i < 2; i++)
        {
            const __m256i aa = _mm256_and_si256(
                _mm256_srli_epi64(a, 16*(1-i)),
                mask
            );
            const __m256i bb = _mm256_and_si256(
                _mm256_srli_epi64(b, 16*(1-i)),
                mask
            );
#ifdef VPC
            const __m256i prod = _mm256_blend_epi32(
                _mm256_shuffle_epi32(
                    _mm256_clmulepi64_epi128(
                        aa,
                        bb,
                        0x11
                    ),
                    0x8D
                ),
                _mm256_shuffle_epi32(
                    _mm256_clmulepi64_epi128(
                        aa,
                        bb,
                        0x00
                    ),
                    0xD8
                ),
                0x33
            );
#else
            const __m256i prod = _mm256_set_m128i(
                /* hi */
                _mm_blend_epi32(
                    _mm_shuffle_epi32(
                        _mm_clmulepi64_si128(
                            _mm256_extractf128_si256(aa, 1),
                            _mm256_extractf128_si256(bb, 1),
                            0x11
                        ),
                        0x8D
                    ),
                    _mm_shuffle_epi32(
                        _mm_clmulepi64_si128(
                            _mm256_extractf128_si256(aa, 1),
                            _mm256_extractf128_si256(bb, 1),
                            0x00
                        ),
                        0xD8
                    ),
                    0x3
                ),
                /* lo */
                _mm_blend_epi32(
                    _mm_shuffle_epi32(
                        _mm_clmulepi64_si128(
                            _mm256_extractf128_si256(aa, 0),
                            _mm256_extractf128_si256(bb, 0),
                            0x11
                        ),
                        0x8D
                    ),
                    _mm_shuffle_epi32(
                        _mm_clmulepi64_si128(
                            _mm256_extractf128_si256(aa, 0),
                            _mm256_extractf128_si256(bb, 0),
                            0x00
                        ),
                        0xD8
                    ),
                    0x3
                )
            );
#endif
            hi = _mm256_or_si256(
                _mm256_slli_epi64(hi, 16),
                _mm256_srli_epi32(prod, 16)
            );
            lo = _mm256_or_si256(
                _mm256_slli_epi64(lo, 16),
                _mm256_and_si256(prod, mask)
            );
        }

        const __m256i tmp = _mm256_xor_si256(
            hi,
            _mm256_xor_si256(
                _mm256_srli_epi16(hi, 14),
                _mm256_xor_si256(
                    _mm256_srli_epi16(hi, 13),
                    _mm256_srli_epi16(hi, 11)
                )
            )
        );

        const __m256i rem = _mm256_xor_si256(
            tmp,
            _mm256_xor_si256(
                _mm256_slli_epi16(tmp, 2),
                _mm256_xor_si256(
                    _mm256_slli_epi16(tmp, 3),
                    _mm256_slli_epi16(tmp, 5)
                )
            )
        );

        return _mm256_xor_si256(rem, lo);
    }
#endif

    inline uint64_t packed_rem(uint64_t a)
    {
        uint64_t pack_mask = ff_util::gf_mask | (ff_util::gf_mask << 32);
        uint64_t lo = a & pack_mask;
        uint64_t hi = (a >> gf_n) & pack_mask;

        uint64_t r = hi;

        uint64_t m = 0b11 | (0b11ull << 32);
        r ^= (hi >> 14) & m;
        m = 0b111 | (0b111ull << 32);
        r ^= (hi >> 13) & m;
        m = 0b11111 | (0b11111ull << 32);
        r ^= (hi >> 11) & m;
        r &= pack_mask;
        r ^= (r << 2) ^ (r << 3) ^ (r << 5);
        r &= pack_mask;

        return r ^ lo;
    }

    /* carryless multiplication of a and b, polynomial multiplicatoin that is
     * done with Intel CLMUL
     */
    inline uint64_t clmul(uint64_t a, uint64_t b)
    {
        const __m128i prod = _mm_clmulepi64_si128(
            _mm_set_epi64x(0, a),
            _mm_set_epi64x(0, b),
            0x0
        );

        uint64_t lo = _mm_extract_epi64(prod, 0x0);
        /* discard hi, only support up to 32 bit */
        return lo;
    }

    /* returns s s.t. for some t: s*a + t*field.mod = gcd(field.mod, a)
     * <=> s*a + t*field.mod = 1 taking mod field.mod we get
     * s*a = 1 mod field.mod and thus a^-1 = s mod field.mod*/
    inline uint64_t ext_euclid(uint64_t a)
    {
        // assert(a != 0)
        uint64_t s = 0x1;
        uint64_t s_next = 0x0;
        uint64_t r = a;
        uint64_t r_next = ff_util::gf_mod;
        uint64_t tmp;

        while (r_next != 0x0)
        {
            uint64_t q = ff_util::quo(r, r_next);
            tmp = r ^ ff_util::clmul(q, r_next);
            r = r_next;
            r_next = tmp;

            tmp = s ^ ff_util::clmul(q, s_next);
            s = s_next;
            s_next = tmp;
        }

        return s;
    }

    inline uint64_t packed_clmul(uint64_t a, uint64_t b)
    {
        __m128i aa = _mm_set_epi64x(a >> 32, a & ff_util::gf_mask);
        __m128i bb = _mm_set_epi64x(b >> 32, b & ff_util::gf_mask);
        __m128i prod[2];
        prod[0] = _mm_clmulepi64_si128(aa, bb, 0x00);
        prod[1] = _mm_clmulepi64_si128(aa, bb, 0x11);

        uint64_t res = _mm_extract_epi64(prod[0], 0x0);
        res |= _mm_extract_epi64(prod[1], 0x0) << 32;

        return res;
    }
}

class GF_element
{
private:
    uint64_t repr;

public:
    GF_element() { }

    GF_element(const uint64_t n)
    {
        this->repr = n;
    }

    GF_element(const GF_element &e)
    {
        this->repr = e.get_repr();
    }

    GF_element operator+(const GF_element &other) const
    {
        return GF_element(this->repr ^ other.get_repr());
    }

    GF_element &operator+=(const GF_element &other)
    {
        this->repr ^= other.get_repr();
        return *this;
    }

    GF_element operator*(const GF_element &other) const
    {
        const uint64_t prod = ff_util::clmul(
            this->repr,
            other.get_repr()
        );

        return GF_element(
            ff_util::rem(prod)
        );
    }

    GF_element &operator*=(const GF_element &other)
    {
        const uint64_t prod = ff_util::clmul(
            this->repr,
            other.get_repr()
        );

        this->repr = ff_util::rem(prod);

        return *this;
    }

    GF_element inv() const
    {
        return GF_element(ff_util::ext_euclid(this->repr));
    }

    void inv_in_place()
    {
        this->repr = ff_util::ext_euclid(this->repr);
    }

    GF_element operator/(const GF_element &other) const
    {
        return *this * other.inv();
    }

    GF_element &operator/=(const GF_element &other)
    {
        const uint64_t inv = ff_util::ext_euclid(other.get_repr());
        const uint64_t prod = ff_util::clmul(
            this->repr,
            inv
            );

        this->repr = ff_util::rem(prod);
        return *this;
    }

    bool operator==(const GF_element &other) const
    {
        return this->repr == other.get_repr();
    }

    uint64_t get_repr() const { return this->repr; }

    GF_element operator-(const GF_element &other) const
    {
        return *this + other;
    }

    GF_element &operator-=(const GF_element &other)
    {
        this->repr ^= other.get_repr();
        return *this;
    }

    GF_element &operator=(const GF_element &other)
    {
        this->repr = other.get_repr();
        return *this;
    }

    bool operator!=(const GF_element &other) const
    {
        return !(*this == other);
    }

    bool operator>(const GF_element &other) const
    {
        return this->repr > other.get_repr();
    }

    void print() const
    {
        std::cout << std::bitset<16>(this->repr) << " ";
    }
};

/* GF(2^n) */
class GF2n
{
private:
    uint64_t mask;
    int n;
    uint64_t mod;

public:
    GF2n() {}

    void init()
    {
        this->n = ff_util::gf_n;
        /* x^16 + x^5 + x^3 + x^2 +  1 */
        this->mod = ff_util::gf_mod;
        this->mask = ff_util::gf_mask;

        if (global::output)
        {
            std::cout << "initialized GF(2^" << this->n << ") with modulus: ";
            for (int i = n; i >= 0; i--)
            {
                if ((this->mod >> i) & 1)
                    std::cout << "1";
                else
                    std::cout << "0";
            }
            std::cout << std::endl;
        }
    }

    int get_n() const { return this->n; }
    uint64_t get_mod() const { return this->mod; }
    uint64_t get_mask() const { return this->mask; }

    GF_element zero() const { return GF_element(0); }
    GF_element one() const { return GF_element(1); }
    GF_element random() const
    {
        return GF_element(global::randgen() & this->mask);
    }

};
#endif
