/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#include <iostream>
#include <getopt.h>

#include "../../src/global.hh"
#include "../../src/gf.hh"

#include "gf_test.hh"
#include "fmatrix_test.hh"

util::rand64bit global::randgen;
GF2n global::F;
bool global::output = false;

using namespace std;

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        cout << "-g for GF tests" << endl;
        cout << "-m for FMatrix tests" << endl;
        cout << "-d $int dimension of matrix" << endl;
        cout << "-t $int how many times random tests are done" << endl;
        cout << "-s $int for seed to pseudo random generator" << endl;
        return 0;
    }

    bool gft = false;
    bool mt = false;
    int dim = 10;
    int tests = 10000;
    int opt;
    uint64_t seed = time(nullptr);

    while ((opt = getopt(argc, argv, "gmd:t:s:")) != -1)
    {
        switch (opt)
        {
        case 's':
            seed = stoll(optarg);
            break;
        case 'd':
            dim = stoi(optarg);
            break;
        case 'g':
            gft = true;
            break;
        case 'm':
            mt = true;
            break;
        case 't':
            tests = stoi(optarg);
            break;
        }
    }

    cout << "seed: " << seed << endl;
    global::randgen.init(seed);
    global::F.init();

    if (gft)
        GF_test f;
    if (mt)
        FMatrix_test fm(dim, tests);

    return 0;
}
