/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#ifndef FMATRIX_H
#define FMATRIX_H

#include <iostream>
#include <valarray>

#include "gf.hh"

class FMatrix
{
private:
    std::valarray<GF_element> m;
    int n;

public:
    FMatrix() { }

    FMatrix(int n, std::valarray<GF_element> &matrix): m(n*n)
    {
        this->n = n;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                m[i*n + j] = matrix[i*n + j];
    }

    /* uses gaussian elimination with pivoting.
     * modifies the object it is called on. */
    GF_element det()
    {
        GF_element det = global::F.one();
        for (int col = 0; col < this->n; col++)
        {
            /* pivot */
            GF_element mx = global::F.zero();
            int mxi = -1;
            for (int row = col; row < this->n; row++)
            {
                if (this->operator()(row,col) != global::F.zero())
                {
                    mx = this->operator()(row,col);
                    mxi = row;
                    break;
                }
            }

            if (mx == global::F.zero())
                return global::F.zero();

            if (mxi != col)
                this->swap_rows(mxi, col);

            det *= mx;
            mx.inv_in_place();
            this->mul_row(col, mx);
#ifdef PAR
            #pragma omp parallel for
#endif
            for (int row = col+1; row < this->n; row++)
                this->row_op(col, row, this->operator()(row,col));
        }
        return det;
    }

    /* returns a copy of this */
    FMatrix copy() const
    {
        std::valarray<GF_element> m(this->n * this->n);

        for (int row = 0; row < this->n; row++)
            for (int col = 0; col < this->n; col++)
                m[row*n + col] = this->operator()(row,col);

        return FMatrix(this->n, m);
    }

    const std::valarray<GF_element> &get_m() const { return this->m; }

    int get_n() const { return this->n; }

    void mul(int row, int col, const GF_element &v)
    {
        this->m[row*this->n + col] *= v;
    }

    void mul_row(int row, const GF_element &v)
    {
        for (int col = 0; col < this->n; col++)
            this->m[row*this->n + col] *= v;
    }

    /* subtract v times r1 from r2 */
    void row_op(int r1, int r2, GF_element v)
    {
        for (int col = 0; col < this->n; col++)
            this->m[r2*this->n + col] -= v*this->operator()(r1,col);
    }

    const GF_element &operator()(int row, int col) const
    {
        return this->m[row*this->n + col];
    }

    bool operator==(const FMatrix &other) const
    {
        if (this->n != other.get_n())
            return false;

        for (int i = 0; i < this->n; i++)
            for (int j = 0; j < this->n; j++)
                if (this->operator()(i,j) != other(i,j))
                    return false;

        return true;
    }

    bool operator!=(const FMatrix &other) const
    {
        return !(*this == other);
    }

    void set(int row, int col, GF_element val)
    {
        this->m[row*this->n + col] = val;
    }

    /* swap rows r1 and r2 starting from column idx */
    void swap_rows(int r1, int r2, int idx = 0)
    {
        GF_element tmp;
        for (int col = idx; col < this->n; col++)
        {
            tmp = this->operator()(r1, col);
            this->set(r1, col, this->operator()(r2,col));
            this->set(r2, col, tmp);
        }
    }

    void print() const
    {
        for (int row = 0; row < this->n; row++)
        {
            for (int col = 0; col < this->n; col++)
                this->operator()(row, col).print();
            std::cout << std::endl;
        }
    }
};

namespace util
{
    inline FMatrix random_fmatrix(int dim)
    {
        std::valarray<GF_element> m(dim*dim);
        for (int i = 0; i < dim*dim; i++)
            m[i] = global::F.random();
        return FMatrix(dim, m);
    }
}

#endif
