#ifndef BRUTEFORCEALGORITHM_H
#define BRUTEFORCEALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>


class BruteForceAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceAlgorithm() {}

        virtual ~BruteForceAlgorithm() {}


        unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const override
        {
            int numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_container_t> pNeighbors = this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);


            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighbors));
        }
    protected:

    private:
};

#endif // BRUTEFORCEALGORITHM_H
