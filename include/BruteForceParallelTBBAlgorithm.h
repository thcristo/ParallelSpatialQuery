#ifndef BRUTEFORCEPARALLELTBBALGORITHM_H
#define BRUTEFORCEPARALLELTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <tbb/tbb.h>

using namespace tbb;

class BruteForceParallelTBBAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceParallelTBBAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }

        virtual ~BruteForceParallelTBBAlgorithm() {}

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            int numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            typedef blocked_range<point_vector_t::const_iterator> point_range_t;

            auto start = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            task_scheduler_init scheduler(task_scheduler_init::deferred);

            if (numThreads > 0)
            {
                scheduler.initialize(numThreads);
            }
            else
            {
                scheduler.initialize(task_scheduler_init::automatic);
            }

            parallel_for(point_range_t(inputDatasetBegin, inputDatasetEnd), [&](point_range_t& range)
                {
                    auto rangeBegin = range.begin();
                    auto rangeEnd = range.end();

                    for (auto inputPoint = rangeBegin; inputPoint < rangeEnd; ++inputPoint)
                    {
                        auto& neighbors = pNeighborsContainer->at(inputPoint->id - 1);

                        for (auto trainingPoint = trainingDatasetBegin; trainingPoint < trainingDatasetEnd; ++trainingPoint)
                        {
                            AddNeighbor(inputPoint, trainingPoint, neighbors);
                        }
                    }
                }
            );

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(problem, "bruteforce_parallel_tbb", pNeighborsContainer, elapsed, chrono::duration<double>()));
        }

    private:
        int numThreads;

};

#endif // BRUTEFORCEPARALLELTBBALGORITHM_H
