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
        unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const  override
        {
            int numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_container_t> pNeighbors = this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighbors));
        }

    protected:

    private:
};

#endif // PARALLELPLANESWEEPALGORITHM_H
