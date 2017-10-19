#ifndef BRUTEFORCEALGORITHM_H
#define BRUTEFORCEALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>


class BruteForceAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceAlgorithm() {}

        virtual ~BruteForceAlgorithm() {}


        AllKnnResult* Process(const AllKnnProblem& problem) const override
        {
            int numNeighbors = problem.GetNumNeighbors();

            neighbors_container_t* pNeighbors = this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);
            /*
            for (int i=0; i < inputPointsNum; ++i)
            {
                for (int j=0; j < trainingPointsNum; ++j)
                {


                }
            }
            */

            return new AllKnnResult();
        }
    protected:

    private:
};

#endif // BRUTEFORCEALGORITHM_H
