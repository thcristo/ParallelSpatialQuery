#ifndef PARALLELPLANESWEEPALGORITHM_H
#define PARALLELPLANESWEEPALGORITHM_H

#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "AbstractAllKnnAlgorithm.h"

class ParallelPlaneSweepAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        ParallelPlaneSweepAlgorithm();
        virtual ~ParallelPlaneSweepAlgorithm();
        AllKnnResult* Process(const AllKnnProblem& problem) const  override;

    protected:

    private:
};

#endif // PARALLELPLANESWEEPALGORITHM_H
