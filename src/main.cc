/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#include <iostream>
#include <vector>
#include <cmath>
#include <getopt.h>
#include <omp.h>

#include "global.hh"
#include "gf.hh"
#include "fmatrix.hh"
#include "packed_fmatrix.hh"

using namespace std;

util::rand64bit global::randgen;
GF2n global::F;
bool global::output = true;

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        cout << "-d matrix dimension" << endl;
        cout << "-n number of matrices" << endl;
        cout << "-b compute baseline speed" << endl;
        cout << "-s for seed" << endl;
        return 0;
    }

    int opt;

    bool baseline = false;
    uint64_t seed = time(nullptr);
    uint64_t dim = 100;
    uint64_t n = 10;

    while ((opt = getopt(argc, argv, "bd:n:s:")) != -1)
    {
        switch (opt)
        {
        case 'b':
            baseline = true;
            break;
        case 's':
            seed = stoi(optarg);
            break;
        case 'd':
            dim = stoi(optarg);
            break;
        case 'n':
            n = stoi(optarg);
            break;
        case '?':
            cout << "call with no arguments for help" << endl;
            return -1;
        }
    }

    cout << "seed: " << seed << endl;
    global::randgen.init(seed);

    global::F.init();

    vector<GF_element> dets(n);
    vector<FMatrix> matrices(n);
    vector<Packed_FMatrix> packed_matrices(n);
    for (uint64_t i = 0; i < n; i++)
    {
        matrices[i] = util::random_fmatrix(dim);
        packed_matrices[i] = Packed_FMatrix(matrices[i]);
    }

    cout << "computing determinant of " << n << " matrices with dimensions "
         << dim << "x" << dim << "." << endl;

    double start;
    double end;

    if (baseline)
    {
        start = omp_get_wtime();
        for (uint64_t i = 0; i < n; i++)
            dets[i] = matrices[i].det();
        end = omp_get_wtime();

        cout << "computed determinants in " << end - start << " seconds." << endl;
    }

    start = omp_get_wtime();
    for (uint64_t i = 0; i < n; i++)
        dets[i] = packed_matrices[i].det();
    end = omp_get_wtime();
    cout << "computed packed determinants in " << end - start
         << " seconds." << endl;

    return 0;
}
