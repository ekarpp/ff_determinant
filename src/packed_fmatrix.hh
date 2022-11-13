/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#ifndef P_FMATRIX_H
#define P_FMATRIX_H

#include <valarray>
#include <vector>
#include <immintrin.h>

#include "global.hh"
#include "gf.hh"
#include "fmatrix.hh"



#ifndef PAR
#define PAR 0
#endif

#ifdef AVX512
#define VECTOR_N 32
typedef long long avx_t __attribute__ ((vector_size (64)));
#else
#define VECTOR_N 16
typedef long long avx_t __attribute__ ((vector_size (32)));
#endif

#define MOVE_ID_VEC(index)                              \
    {                                                   \
        if (index%2)                                    \
            onepos = _mm256_slli_epi32(                 \
                _mm256_permutevar8x32_epi32(            \
                    onepos,                             \
                    permute_idx                         \
                ),                                      \
                16                                      \
            );                                          \
        else                                            \
            onepos = _mm256_srli_epi32(onepos, 16);     \
    }

#ifdef AVX512
#define DET_LOOP(index)                                         \
    {                                                           \
        int r0 = VECTOR_N*col + index;                          \
        avx_t mx;                                             \
        int mxi = -1;                                           \
        const avx_t one = _mm512_set1_epi8(0xFF);               \
        for (int row = r0; row < this->rows; row++)             \
        {                                                       \
            const uint32_t mask = _mm512_test_epi16_mask(       \
                this->get(row,col),                             \
                one                                             \
            );                                                  \
            if (((mask >> (VECTOR_N - 1 - index)) & 1) == 1)    \
            {                                                   \
                mx = this->get(row,col);                        \
                mxi = row;                                      \
                break;                                          \
            }                                                   \
        }                                                       \
        if (mxi == -1)                                          \
            return global::F.zero();                            \
        const char mask = (VECTOR_N - 1 - index) / 2;           \
        const avx_t idx = _mm512_set1_epi16(mask);              \
        uint64_t mx_ext = _mm512_cvtsi512_si32(                 \
            _mm512_permutexvar_epi16(                           \
                mx,                                             \
                idx                                             \
            )                                                   \
        ) & ff_util::gf_mask;                                   \
        if (mxi != r0)                                          \
            this->swap_rows(mxi, r0);                           \
        /* vectorize? */                                        \
        det = ff_util::rem(                                     \
            ff_util::clmul(det, mx_ext)                         \
        );                                                      \
        mx_ext = ff_util::ext_euclid(mx_ext);                   \
        this->mul_row(r0, mx_ext);                              \
        /* vectorize end? */                                    \
        if (PAR) {                                              \
            _Pragma("omp parallel for")                         \
            for (int row = r0 + 1; row < this->rows; row++)     \
                this->row_op(r0, row,                           \
                             _mm512_permutexvar_epi16(          \
                                 this->get(row, col),           \
                                 idx                            \
                             )                                  \
                );                                              \
        } else {                                                \
            for (int row = r0 + 1; row < this->rows; row++)     \
                this->row_op(r0, row,                           \
                             _mm512_permutexvar_epi16(          \
                                 this->get(row, col),           \
                                 idx                            \
                             )                                  \
                );                                              \
        }                                                       \
    }
#else
#define ELIM_LOOP(index)                                            \
        for (int row = r0 + 1; row < this->rows; row++)             \
        {                                                           \
            avx_t val = _mm256_permutevar8x32_epi32(              \
                this->get(row, col),                                \
                idx                                                 \
            );                                                      \
            if (index % 2)                                          \
                val = _mm256_blend_epi16(                           \
                    val,                                            \
                    _mm256_slli_epi32(val, 16),                     \
                    0xAA                                            \
                );                                                  \
            else                                                    \
                val = _mm256_blend_epi16(                           \
                    _mm256_srli_epi32(val, 16),                     \
                    val,                                            \
                    0xAA                                            \
                );                                                  \
            this->row_op(r0, row, val);                             \
        }

#define DET_LOOP(index)                                         \
    {                                                           \
        int r0 = VECTOR_N*col + index;                          \
        avx_t mx;                                             \
        int mxi = -1;                                           \
        avx_t cmpmsk = _mm256_set_epi64x(                     \
            global::F.get_mask() << 48,                         \
            0,                                                  \
            0,                                                  \
            0                                                   \
        );                                                      \
        if (index >= (VECTOR_N / 2))                            \
            cmpmsk = _mm256_permute4x64_epi64(cmpmsk, 0x0C);    \
        cmpmsk = _mm256_srli_si256(cmpmsk, 2 * (index % 8));    \
        for (int row = r0; row < this->rows; row++)             \
        {                                                       \
            char ZF = _mm256_testz_si256(                       \
                cmpmsk,                                         \
                this->get(row,col)                              \
            );                                                  \
            if (ZF == 0)                                        \
            {                                                   \
                mx = this->get(row,col);                        \
                mxi = row;                                      \
                break;                                          \
            }                                                   \
        }                                                       \
        if (mxi == -1)                                          \
            return global::F.zero();                            \
        uint64_t mx_ext =                                       \
            _mm256_extract_epi16(mx, VECTOR_N - 1 - index);     \
        if (mxi != r0)                                          \
            this->swap_rows(mxi, r0);                           \
        /* vectorize? */                                        \
        det = ff_util::rem(                                     \
            ff_util::clmul(det, mx_ext)                         \
        );                                                      \
        mx_ext = ff_util::ext_euclid(mx_ext);                   \
        this->mul_row(r0, mx_ext);                              \
        char mask = (VECTOR_N - 1 - index) / 2;                 \
        avx_t idx = _mm256_set1_epi32(mask);                  \
        /* vectorize end? */                                    \
        if (PAR) {                                              \
            _Pragma("omp parallel for")                         \
            ELIM_LOOP(index);                                   \
        } else {                                                \
            ELIM_LOOP(index);                                   \
        }                                                       \
    }
#endif

class Packed_FMatrix
{
private:
    int rows;
    int cols;
    // original matrix n moduloe VECTOR_N
    int nmod;
    std::vector<avx_t> m;

    avx_t get(int row, int col) const
    {
        return this->m[row*this->cols + col];
    }

    void set(int row, int col, avx_t v)
    {
        this->m[row*this->cols + col] = v;
    }

public:
    Packed_FMatrix() {}

    Packed_FMatrix(
        const FMatrix &matrix
    )
    {
        this->nmod = matrix.get_n() % VECTOR_N;
        this->rows = matrix.get_n();
        if (this->nmod)
            this->rows += VECTOR_N - this->nmod;
        this->cols = this->rows / VECTOR_N;

        this->m.resize(this->rows * this->cols);

        for (int r = 0; r < matrix.get_n(); r++)
        {
            for (int c = 0; c < matrix.get_n() / VECTOR_N; c++)
                this->set(r, c,
#ifdef AVX512
                          _mm512_set_epi16(
#else
                          _mm256_set_epi16(
#endif
                              matrix(r, VECTOR_N*c +  0).get_repr(),
                              matrix(r, VECTOR_N*c +  1).get_repr(),
                              matrix(r, VECTOR_N*c +  2).get_repr(),
                              matrix(r, VECTOR_N*c +  3).get_repr(),
                              matrix(r, VECTOR_N*c +  4).get_repr(),
                              matrix(r, VECTOR_N*c +  5).get_repr(),
                              matrix(r, VECTOR_N*c +  6).get_repr(),
                              matrix(r, VECTOR_N*c +  7).get_repr(),
                              matrix(r, VECTOR_N*c +  8).get_repr(),
                              matrix(r, VECTOR_N*c +  9).get_repr(),
                              matrix(r, VECTOR_N*c + 10).get_repr(),
                              matrix(r, VECTOR_N*c + 11).get_repr(),
                              matrix(r, VECTOR_N*c + 12).get_repr(),
                              matrix(r, VECTOR_N*c + 13).get_repr(),
                              matrix(r, VECTOR_N*c + 14).get_repr(),
#ifndef AVX512
                              matrix(r, VECTOR_N*c + 15).get_repr())
#else
                              matrix(r, VECTOR_N*c + 15).get_repr(),
                              matrix(r, VECTOR_N*c + 16).get_repr(),
                              matrix(r, VECTOR_N*c + 17).get_repr(),
                              matrix(r, VECTOR_N*c + 18).get_repr(),
                              matrix(r, VECTOR_N*c + 19).get_repr(),
                              matrix(r, VECTOR_N*c + 20).get_repr(),
                              matrix(r, VECTOR_N*c + 21).get_repr(),
                              matrix(r, VECTOR_N*c + 22).get_repr(),
                              matrix(r, VECTOR_N*c + 23).get_repr(),
                              matrix(r, VECTOR_N*c + 24).get_repr(),
                              matrix(r, VECTOR_N*c + 25).get_repr(),
                              matrix(r, VECTOR_N*c + 26).get_repr(),
                              matrix(r, VECTOR_N*c + 27).get_repr(),
                              matrix(r, VECTOR_N*c + 28).get_repr(),
                              matrix(r, VECTOR_N*c + 29).get_repr(),
                              matrix(r, VECTOR_N*c + 30).get_repr(),
                              matrix(r, VECTOR_N*c + 31).get_repr())
#endif
                    );
            if (this->nmod)
            {
                int c = this->cols - 1;
                uint64_t elems[VECTOR_N];
                for (int i = 0; i < VECTOR_N; i++)
                    elems[i] = 0;
                for (int i = 0; i < this->nmod; i++)
                    elems[i] = matrix(r, VECTOR_N*c + i).get_repr();

                this->set(r, c,
#ifdef AVX512
                          _mm512_set_epi16(
#else
                          _mm256_set_epi16(
#endif
                              elems[0],  elems[1],  elems[2],  elems[3],
                              elems[4],  elems[5],  elems[6],  elems[7],
                              elems[8],  elems[9],  elems[10], elems[11],
#ifndef AVX512
                              elems[12], elems[13], elems[14], elems[15])
#else
                              elems[12], elems[13], elems[14], elems[15],
                              elems[16], elems[17], elems[18], elems[19],
                              elems[20], elems[21], elems[22], elems[23],
                              elems[24], elems[25], elems[26], elems[27],
                              elems[28], elems[29], elems[30], elems[31])
#endif

                );
            }
        }

#ifdef AVX512
        avx_t onepos = _mm512_set_epi64(1ull << 48, 0, 0, 0,
                                        0, 0, 0, 0);
        const avx_t permute_idx = _mm512_set_epi16(
            0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111, 0b000,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        );
        for (int i = 0; i < this->nmod; i++)
            onepos = _mm512_permutexvar_epi16(permute_idx, onepos);

        for (int r = matrix.get_n(); r < this->rows; r++)
        {
            for (int c = 0; c < this->cols - 1; c++)
                this->set(r, c, _mm512_setzero_si512());
            this->set(r, this->cols - 1, onepos);
            onepos = _mm512_permutexvar_epi16(permute_idx, onepos);
        }
#else
        /* cycle all one position to right */
        const avx_t permute_idx = _mm256_set_epi32(
            0b000,
            0b111,
            0b110,
            0b101,
            0b100,
            0b011,
            0b010,
            0b001
        );

        avx_t onepos = _mm256_set_epi64x(1ull << 48, 0, 0, 0);
        for (int i = 0; i < this->nmod; i++)
            MOVE_ID_VEC(i);

        for (int r = matrix.get_n(); r < this->rows; r++)
        {
            for (int c = 0; c < this->cols - 1; c++)
                this->set(r, c, _mm256_setzero_si256());
            this->set(r, this->cols - 1, onepos);
            MOVE_ID_VEC(r % VECTOR_N);
        }
#endif
    }

    void swap_rows(int r1, int r2)
    {
        avx_t tmp;
        for (int c = 0; c < this->cols; c++)
        {
            tmp = this->get(r1, c);
            this->set(r1, c, this->get(r2, c));
            this->set(r2, c, tmp);
        }
    }

    void mul_row(int row, uint64_t v)
    {
#ifdef AVX512
        avx_t pack = _mm512_set1_epi16(v);
#else
        avx_t pack = _mm256_set1_epi16(v);
#endif
        for (int col = 0; col < this->cols; col++)
            this->set(row, col,
                      ff_util::wide_mul(this->get(row, col), pack)
                );
    }

    /* subtract v times r1 from r2 */
    void row_op(int r1, int r2, avx_t pack)
    {
        for (int col = 0; col < this->cols; col++)
        {
            avx_t tmp = ff_util::wide_mul(this->get(r1, col), pack);

            this->set(r2, col,
#ifdef AVX512
                      _mm512_xor_si512(this->get(r2,col), tmp)
#else
                      _mm256_xor_si256(this->get(r2, col), tmp)
#endif
                );
        }
    }

    GF_element det()
    {
        uint64_t det = 0x1;
        for (int col = 0; col < this->cols; col++)
        {
            DET_LOOP(0);  DET_LOOP(1);  DET_LOOP(2);  DET_LOOP(3);
            DET_LOOP(4);  DET_LOOP(5);  DET_LOOP(6);  DET_LOOP(7);
            DET_LOOP(8);  DET_LOOP(9);  DET_LOOP(10); DET_LOOP(11);
            DET_LOOP(12); DET_LOOP(13); DET_LOOP(14); DET_LOOP(15);
            DET_LOOP(16); DET_LOOP(17); DET_LOOP(18); DET_LOOP(19);
            DET_LOOP(20); DET_LOOP(21); DET_LOOP(22); DET_LOOP(23);
            DET_LOOP(24); DET_LOOP(25); DET_LOOP(26); DET_LOOP(27);
            DET_LOOP(28); DET_LOOP(29); DET_LOOP(30); DET_LOOP(31);
        }
        return GF_element(det);
    }
};

#endif
