/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#ifndef P_FMATRIX_H
#define P_FMATRIX_H

#include <valarray>
#include <vector>
#include <immintrin.h>

#include "global.hh"
#include "gf.hh"
#include "fmatrix.hh"

typedef long long int long4_t __attribute__ ((vector_size (32)));

#ifndef PAR
#define PAR 0
#endif

#define VECTOR_N 8

#define COEFF_LOOP(index)                                               \
    {                                                                   \
        for (int col = 0; col < this->cols - 1; col++)                  \
            coeffs[col] = _mm256_blend_epi32(                           \
                _mm256_permutevar8x32_epi32(coeffs[col + 0], idx),      \
                _mm256_permutevar8x32_epi32(coeffs[col + 1], idx),      \
                0xFF >> index                                           \
            );                                                          \
    }

#define DET_LOOP(index)                                         \
    {                                                           \
        int r0 = VECTOR_N*col + index;                          \
        long4_t mx;                                             \
        int mxi = -1;                                           \
        long4_t cmpmsk = _mm256_set_epi64x(                     \
            0xFFFFull << 32,                                    \
            0,                                                  \
            0,                                                  \
            0                                                   \
        );                                                      \
        if (index >= 4)                                         \
            cmpmsk = _mm256_permute4x64_epi64(cmpmsk, 0x0C);    \
        cmpmsk = _mm256_srli_si256(cmpmsk, 4*(index%4));        \
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
            _mm256_extract_epi32(mx, VECTOR_N - 1 - index);     \
        if (mxi != r0)                                          \
            this->swap_rows(mxi, r0);                           \
        /* vectorize? */                                        \
        det = ff_util::rem(                                     \
            ff_util::clmul(det, mx_ext)                         \
        );                                                      \
        mx_ext = ff_util::ext_euclid(mx_ext);                   \
        this->mul_row(r0, mx_ext);                              \
        /* vectorize end? */                                    \
        char mask = VECTOR_N - 1 - index;                       \
        long4_t idx = _mm256_set1_epi32(mask);                  \
        if (PAR) {                                              \
            _Pragma("omp parallel for")                         \
            for (int row = r0 + 1; row < this->rows; row++)     \
            {                                                   \
                long4_t val = _mm256_permutevar8x32_epi32(      \
                    this->get(row, col),                        \
                    idx                                         \
                );                                              \
                this->row_op(r0, row, val);                     \
            }                                                   \
        } else {                                                \
            for (int row = r0 + 1; row < this->rows; row++)     \
            {                                                   \
                long4_t val = _mm256_permutevar8x32_epi32(      \
                    this->get(row, col),                        \
                    idx                                         \
                );                                              \
                this->row_op(r0, row, val);                     \
            }                                                   \
        }                                                       \
    }

class Packed_FMatrix
{
private:
    int rows;
    int cols;
    // original matrix n moduloe VECTOR_N
    int nmod;
    std::vector<long4_t> m;

    long4_t get(int row, int col) const
    {
        return this->m[row*this->cols + col];
    }

    void set(int row, int col, long4_t v)
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
        if (this->rows % VECTOR_N)
            this->rows += VECTOR_N - (matrix.get_n() % VECTOR_N);
        this->cols = this->rows / VECTOR_N;

        this->m.resize(this->rows * this->cols);

        for (int r = 0; r < matrix.get_n(); r++)
        {
            for (int c = 0; c < matrix.get_n() / VECTOR_N; c++)
                this->set(r, c,
                          _mm256_set_epi64x(
                              matrix(r, VECTOR_N*c + 0).get_repr() << 32
                                  | matrix(r, VECTOR_N*c + 1).get_repr(),
                              matrix(r, VECTOR_N*c + 2).get_repr() << 32
                                  | matrix(r, VECTOR_N*c + 3).get_repr(),
                              matrix(r, VECTOR_N*c + 4).get_repr() << 32
                                  | matrix(r, VECTOR_N*c + 5).get_repr(),
                              matrix(r, VECTOR_N*c + 6).get_repr() << 32
                                  | matrix(r, VECTOR_N*c + 7).get_repr()
                          )
                    );
            if (this->nmod)
            {
                int c = this->cols - 1;
                uint64_t elems[4];
                elems[0] = 0; elems[1] = 0; elems[2] = 0; elems[3] = 0;
                for (int i = 0; i < this->nmod; i++)
                    elems[i/2] |= matrix(r, VECTOR_N*c + i).get_repr() << (32*(1 - i%2));

                this->set(r, c, _mm256_set_epi64x(
                              elems[0],
                              elems[1],
                              elems[2],
                              elems[3]
                          )
                );
            }
        }
        for (int r = matrix.get_n(); r < this->rows; r++)
        {
            for (int c = 0; c < this->cols - 1; c++)
                this->set(r, c, _mm256_setzero_si256());

            /* lazy.... */
            switch (r % VECTOR_N)
            {
            case 1:
                this->set(r, this->cols - 1, _mm256_set_epi64x(
                              1, 0, 0, 0
                         )
                );
                break;
            case 2:
                this->set(r, this->cols - 1, _mm256_set_epi64x(
                              0, 1ull << 32, 0, 0
                         )
                );
                break;
            case 3:
                this->set(r, this->cols - 1, _mm256_set_epi64x(
                              0, 1, 0, 0
                         )
                );
                break;
            case 4:
                this->set(r, this->cols - 1, _mm256_set_epi64x(
                              0, 0, 1ull << 32, 0
                         )
                );
                break;
            case 5:
                this->set(r, this->cols - 1, _mm256_set_epi64x(
                              0, 0, 1, 0
                         )
                );
                break;
            case 6:
                this->set(r, this->cols - 1, _mm256_set_epi64x(
                              0, 0, 0, 1ull << 32
                         )
                );
                break;
            case 7:
                this->set(r, this->cols - 1, _mm256_set_epi64x(
                              0, 0, 0, 1
                         )
                );
                break;
            }

        }
    }

    void swap_rows(int r1, int r2)
    {
        long4_t tmp;
        for (int c = 0; c < this->cols; c++)
        {
            tmp = this->get(r1, c);
            this->set(r1, c, this->get(r2, c));
            this->set(r2, c, tmp);
        }
    }

    void mul_row(int row, uint64_t v)
    {
        long4_t pack = _mm256_set1_epi32(v);
        for (int col = 0; col < this->cols; col++)
            this->set(row, col,
                      ff_util::wide_mul(this->get(row, col), pack)
                );
    }

    /* subtract v times r1 from r2 */
    void row_op(int r1, int r2, long4_t pack)
    {
        for (int col = 0; col < this->cols; col++)
        {
            long4_t tmp = ff_util::wide_mul(this->get(r1, col), pack);

            this->set(r2, col,
                      _mm256_xor_si256(this->get(r2, col), tmp)
                );
        }
    }

    GF_element det()
    {
        uint64_t det = 0x1;
        for (int col = 0; col < this->cols; col++)
        {
            DET_LOOP(0);
            DET_LOOP(1);
            DET_LOOP(2);
            DET_LOOP(3);
            DET_LOOP(4);
            DET_LOOP(5);
            DET_LOOP(6);
            DET_LOOP(7);
        }
        return GF_element(det);
    }
};

#endif
