#ifndef BRUTEFORCEPARALLELTBBALGORITHM_H
#define BRUTEFORCEPARALLELTBBALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>
#include <tbb/tbb.h>

using namespace tbb;

class BruteForceParallelTBBAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceParallelTBBAlgorithm() {}
        virtual ~BruteForceParallelTBBAlgorithm() {}

        unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const override
        {
            int numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_container_t> pNeighborsContainer = this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            typedef blocked_range<point_vector_t::const_iterator> point_range_t;

            auto start = chrono::high_resolution_clock::now();

            parallel_for(point_range_t(inputDataset.cbegin(), inputDataset.cend()), [&](point_range_t& range)
                {
                    for (auto inputPoint = range.begin(); inputPoint != range.end(); ++inputPoint)
                    {
                        auto& neighbors = pNeighborsContainer->at(inputPoint->id);

                        for (auto trainingPoint = trainingDataset.cbegin(); trainingPoint != trainingDataset.cend(); ++trainingPoint)
                        {
                            AddNeighbor(inputPoint, trainingPoint, neighbors);
                        }
                    }
                }
            );

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, "bruteforce_parallel_tbb", problem));
        }
    protected:
        unique_ptr<PointNeighbors> CreatePointNeighbors(size_t numNeighbors) const override
        {
            return unique_ptr<PointNeighbors>(new PointNeighborsPriorityQueue(numNeighbors));
        }
    private:
};

#endif // BRUTEFORCEPARALLELTBBALGORITHM_H
