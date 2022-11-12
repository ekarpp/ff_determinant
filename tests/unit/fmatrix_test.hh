/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#ifndef FMATRIX_TEST_H
#define FMATRIX_TEST_H

#include <valarray>

#include "test.hh"
#include "../../src/gf.hh"
#include "../../src/fmatrix.hh"

class FMatrix_test : Test
{
private:
    int dim;

    GF_element det_heap(const FMatrix &m);
    GF_element term(std::valarray<int> &perm, const FMatrix &m);
    void swap(int i1, int i2, std::valarray<int> &perm);

    void test_determinant_vandermonde();
    void test_det_singular();
    void test_determinant_random();
    void test_packed_determinant();
    void test_packed_determinant_singular();

    void run()
    {
        test_determinant_vandermonde();
        test_determinant_random();
        test_det_singular();
        test_packed_determinant();
        test_packed_determinant_singular();
    }

    FMatrix vandermonde();
    FMatrix random(int n);

public:
    FMatrix_test(int dim, int tests);
};

#endif
