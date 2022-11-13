/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#include <iostream>
#include <immintrin.h>

#include "gf_test.hh"
#include "../../src/gf.hh"
#include "../../src/global.hh"

using namespace std;

GF_test::GF_test()
{
    cout << "----------" << endl;
    cout << "TESTING GF" << endl;
    cout << "----------" << endl;

    this->run();
}

void GF_test::test_add_inverse()
{
    cout << "add inverse: ";
    int err = 0;
    uint64_t i = 0;
    while (i <= global::F.get_mask())
    {
        GF_element e(i);
        if (e + e != global::F.zero()
            || e - e != global::F.zero())
            err++;
        i++;
    }
    end_test(err);
}

void GF_test::test_associativity()
{
    cout << "test associativity: ";
    int err = 0;
    for (int i = 0; i < 10000; i++)
    {
        GF_element a = global::F.random();
        GF_element b = global::F.random();
        GF_element c = global::F.random();
        if (a*(b+c) != c*a + b*a)
            err++;
    }
    this->end_test(err);
}

void GF_test::test_mul_id()
{
    cout << "mul with id: ";
    int err = 0;
    uint64_t i = 0;
    while (i <= global::F.get_mask())
    {
        GF_element e(i);
        if (e * global::F.one() != e)
            err++;
        i++;
    }
    this->end_test(err);
}

void GF_test::test_mul_inverse()
{
    cout << "mul with inverse: ";
    int err = 0;
    uint64_t i = 1;
    while (i <= global::F.get_mask())
    {
        GF_element e(i);
        if (e / e != global::F.one())
            err++;
        i++;
    }
    this->end_test(err);
}

void GF_test::test_packed_rem()
{
    cout << "packed rem: ";
    int err = 0;
    uint64_t mask = global::F.get_mask() |
        (global::F.get_mask() << 32);
    for (int i = 0; i < this->tests; i++)
    {
        uint64_t a = global::randgen() & mask;
        uint64_t b = global::randgen() & mask;

        uint64_t ref = ff_util::rem(
            ff_util::clmul(
                (a >> 32) & global::F.get_mask(),
                (b >> 32) & global::F.get_mask()
            )
        );
        ref <<= 32;

        ref |= ff_util::rem(
            ff_util::clmul(
                a & global::F.get_mask(),
                b & global::F.get_mask()
            )
        );

        uint64_t r = ff_util::packed_rem(
            ff_util::packed_clmul(a, b)
        );

        if (ref != r)
            err++;
    }
    this->end_test(err);
}

void GF_test::test_wide_mul()
{
#ifdef AVX512
#define WIDTH 8
#else
#define WIDTH 4
#endif
    cout << "wide mul: ";
    int err = 0;
    for (int i = 0; i < this->tests / WIDTH; i++)
    {
        uint64_t a[WIDTH];
        uint64_t b[WIDTH];
        int64_t prod[WIDTH];

        for (int j = 0; j < WIDTH; j++)
        {
            a[j] = global::randgen();
            b[j] = global::randgen();
            prod[j] = 0;
            for (int k = 0; k < 4; k++)
            {
                prod[j] |= ff_util::rem(
                    ff_util::clmul(
                        (a[j] >> (16*k)) & global::F.get_mask(),
                        (b[j] >> (16*k)) & global::F.get_mask()
                    )
                ) << (16*k);
            }
        }
#ifdef AVX512
        __m512i aa = _mm512_set_epi64(
            a[7], a[6], a[5], a[4],
            a[3], a[2], a[1], a[0]
         );
        __m512i bb = _mm512_set_epi64(
            b[7], b[6], b[5], b[4],
            b[3], b[2], b[1], b[0]
        );
        __m512i pp = ff_util::wide_mul(aa, bb);

        #pragama GCC unroll 32
        for (int i = 0; i < WIDTH; i++)
        {
            const __m128i tmp = _mm512_extract64x2_epi128(
                pp,
                i / 2
            );
            if (prod[i] != _mm_extract_epi64(tmp, i % 2))
                err++;
        }
#else
        __m256i aa = _mm256_set_epi64x(a[3], a[2], a[1], a[0]);
        __m256i bb = _mm256_set_epi64x(b[3], b[2], b[1], b[0]);
        __m256i pp = ff_util::wide_mul(aa, bb);

        if (prod[0] != _mm256_extract_epi64(pp, 0))
            err++;

        if (prod[1] != _mm256_extract_epi64(pp, 1))
            err++;

        if (prod[2] != _mm256_extract_epi64(pp, 2))
            err++;

        if (prod[3] != _mm256_extract_epi64(pp, 3))
            err++;
#endif
    }
    this->end_test(err);
}
