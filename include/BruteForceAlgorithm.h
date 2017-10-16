#ifndef BRUTEFORCEALGORITHM_H
#define BRUTEFORCEALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>


class BruteForceAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceAlgorithm();
        virtual ~BruteForceAlgorithm();
        AllKnnResult* Process(const AllKnnProblem& problem) const;
    protected:

    private:
};

#endif // BRUTEFORCEALGORITHM_H
