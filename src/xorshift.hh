/* Copyright 2022 Eetu Karppinen. Subject to the MIT license. */
#ifndef XORSHIFT_H
#define XORSHIFT_H

#include <stdint.h>

/* xorshift128+ random value generator */
class Xorshift
{
private:
    uint64_t state[2];

public:
    Xorshift() { }

    void init(uint64_t seed)
    {
        this->state[0] = seed;
        this->state[1] = seed;
    }

    uint64_t next()
    {
        uint64_t tmp = this->state[0];
        this->state[0] = this->state[1];
        tmp ^= tmp << 23;
        tmp ^= tmp >> 18;
        tmp ^= this->state[0] ^ (this->state[0] >> 5);
        this->state[1] = tmp;
        return this->state[0] + this->state[1];
    }
};

#endif
