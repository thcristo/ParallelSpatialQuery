#ifndef PARALLELPLANESWEEPALGORITHM_H
#define PARALLELPLANESWEEPALGORITHM_H

#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "AbstractAllKnnAlgorithm.h"
#include <chrono>

class ParallelPlaneSweepAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        ParallelPlaneSweepAlgorithm();
        virtual ~ParallelPlaneSweepAlgorithm();
        unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const  override
        {
            int numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_container_t> pNeighbors = this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

            auto start = chrono::high_resolution_clock::now();


            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighbors, elapsed, "planesweep_parallel"));
        }

    protected:

    private:
};

#endif // PARALLELPLANESWEEPALGORITHM_H
