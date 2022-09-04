#pragma once

class Rng
{
private:
    const unsigned int multiplier;
    const unsigned long max;
    long rand;

    inline long nextRandonLong(long seed);

public:
    Rng(unsigned long seed);
    virtual void seed(const unsigned long seed);
    virtual double random();
};
