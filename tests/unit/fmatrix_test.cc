/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#include <iostream>
#include <valarray>

#include "fmatrix_test.hh"
#include "../../src/global.hh"
#include "../../src/fmatrix.hh"
#include "../../src/gf.hh"
#include "../../src/packed_fmatrix.hh"

using namespace std;

FMatrix_test::FMatrix_test(int dim, int tests = 0)
{
    if (tests)
        this->tests = tests;
    this->dim = dim;

    cout << "---------------" << endl;
    cout << "TESTING FMATRIX" << endl;
    cout << "---------------" << endl;

    this->run();
}

FMatrix FMatrix_test::vandermonde()
{
    valarray<GF_element> m(this->dim * this->dim);
    uint64_t v = ff_util::rem(global::randgen());

    for (int row = 0; row < this->dim; row++)
    {
        const GF_element e = GF_element(v);
        v = ff_util::rem(v + 2);
        GF_element prod = global::F.one();

        for (int col = 0; col < this->dim; col++)
        {
            m[row*this->dim + col] = prod;
            prod *= e;
        }
    }

    return FMatrix(this->dim, m);
}

GF_element FMatrix_test::term(valarray<int> &perm, const FMatrix &m)
{
    GF_element ret = global::F.one();
    for (int col = 0; col < m.get_n(); col++)
        ret *= m(perm[col], col);
    return ret;
}

void FMatrix_test::swap(int i1, int i2, valarray<int> &perm)
{
    int tmp = perm[i1];
    perm[i1] = perm[i2];
    perm[i2] = tmp;
}

GF_element FMatrix_test::det_heap(const FMatrix &m)
{
    GF_element det = global::F.zero();

    int n = m.get_n();

    /* iterative heaps algo for permutations. compute
     * determinant with the Leibniz formula */
    valarray<int> c(0, n);
    valarray<int> perm(0, n);
    for (int i = 0; i < n; i++)
        perm[i] = i;
    GF_element tt = this->term(perm, m);
    det += tt;

    int i = 0;
    while (i < n)
    {
        if (c[i] < i)
        {
            if (i%2 == 0)
                this->swap(0, i, perm);
            else
                this->swap(c[i], i, perm);
            tt = this->term(perm, m);
            det += tt;
            c[i]++;
            i = 0;
        }
        else
        {
            c[i] = 0;
            i++;
        }
    }

    return det;
}

FMatrix FMatrix_test::random(int n = 0)
{
    if (!n)
        n = this->dim;
    valarray<GF_element> m(n * n);

    for (int row = 0; row < n; row++)
        for (int col = 0; col < n; col++)
            m[row*n + col] = global::F.random();

    return FMatrix(n, m);
}

void FMatrix_test::test_determinant_vandermonde()
{
    cout << "determinant vandermonde: ";
    int err = 0;
    for (int t = 0; t < this->tests; t++)
    {
        FMatrix vander = this->vandermonde();
        GF_element det = global::F.one();

        for (int i = 0; i < this->dim; i++)
            for (int j = i+1; j < this->dim; j++)
                det *= vander(j, 1) - vander(i, 1);

        if (det != vander.det())
            err++;
    }
    end_test(err);
}

void FMatrix_test::test_determinant_random()
{
    cout << "determinant random: ";
    int err = 0;
    for (int t = 0; t < this->tests; t++)
    {
        FMatrix m = this->random(5);
        GF_element d = this->det_heap(m);
        if (d != m.det())
            err++;
    }
    end_test(err);
}

void FMatrix_test::test_det_singular()
{
    cout << "determinant on singular matrices: ";
    int err = 0;
    for (int t = 0; t < this->tests; t++)
    {
        FMatrix m = this->random();
        int r1 = global::randgen() % this->dim;
        int r2 = global::randgen() % this->dim;
        while (r1 == r2)
            r2 = global::randgen() % this->dim;

        /* make it singular */
        for (int col = 0; col < this->dim; col++)
            m.set(r1, col, m(r2, col));

        if (m.det() != global::F.zero())
            err++;
    }
    end_test(err);
}

void FMatrix_test::test_packed_determinant()
{
    cout << "determinant on packed matrices: ";
    int err = 0;
    for (int t = 0; t < this->tests; t++)
    {
        FMatrix m = this->random();
        GF_element pack = Packed_FMatrix(m).det();
        GF_element ref = m.det();

        if (pack != ref)
            err++;
    }
    end_test(err);
}

void FMatrix_test::test_packed_determinant_singular()
{
    cout << "determinant on packed singular matrices: ";
    int err = 0;
    for (int t = 0; t < this->tests; t++)
    {
        FMatrix m = this->random();
        int r1 = global::randgen() % this->dim;
        int r2 = global::randgen() % this->dim;
        while (r1 == r2)
            r2 = global::randgen() % this->dim;

        /* make it singular */
        for (int col = 0; col < this->dim; col++)
            m.set(r1, col, m(r2, col));

        GF_element pack = Packed_FMatrix(m).det();
        GF_element ref = m.det();

        if (pack != ref)
            err++;
    }
    end_test(err);
}
