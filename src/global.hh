/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#ifndef GLOBAL_H
#define GLOBAL_H

#include "xorshift.hh"

class GF2n;

namespace util
{
    class rand64bit
    {
    private:
        Xorshift gen;
    public:
        rand64bit() {}
        void init(uint64_t seed) { this->gen.init(seed); }
        uint64_t operator() () { return this->gen.next(); }
    };

    inline int log2(uint64_t a)
    {
        return 63 - __builtin_clzl(a);
    }
}

namespace global
{
    /* these are defined in main.cc */
    extern util::rand64bit randgen;
    extern GF2n F;
    extern bool output;
}


#endif
