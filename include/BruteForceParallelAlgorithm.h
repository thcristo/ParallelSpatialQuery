#ifndef BRUTEFORCEPARALLELALGORITHM_H
#define BRUTEFORCEPARALLELALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>


class BruteForceParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceParallelAlgorithm() {}
        virtual ~BruteForceParallelAlgorithm() {}

        unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const override
        {
            int numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_container_t> pNeighborsContainer = this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            auto start = chrono::high_resolution_clock::now();

            #pragma omp parallel for
            for (auto inputPoint = inputDataset.cbegin(); inputPoint < inputDataset.cend(); ++inputPoint)
            {
                auto& neighbors = pNeighborsContainer->at(inputPoint->id);

                for (auto trainingPoint = trainingDataset.cbegin(); trainingPoint != trainingDataset.cend(); ++trainingPoint)
                {
                    AddNeighbor(inputPoint, trainingPoint, neighbors);
                }
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, "bruteforce_parallel", problem));
        }
    protected:
        unique_ptr<PointNeighbors> CreatePointNeighbors(size_t numNeighbors) const override
        {
            return unique_ptr<PointNeighbors>(new PointNeighborsPriorityQueue(numNeighbors));
        }
    private:
};

#endif // BRUTEFORCEPARALLELALGORITHM_H
