#ifndef BRUTEFORCEPARALLELTBBALGORITHM_H
#define BRUTEFORCEPARALLELTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <tbb/tbb.h>

using namespace tbb;

template<class ProblemT, class ResultT, class PointVectorT, class PointVectorIteratorT, class NeighborsContainerT>
class BruteForceParallelTBBAlgorithm : public AbstractAllKnnAlgorithm<ProblemT, ResultT, PointVectorT, PointVectorIteratorT>
{
    public:
        BruteForceParallelTBBAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }

        virtual ~BruteForceParallelTBBAlgorithm() {}

        string GetTitle() const
        {
            return "Brute force parallel TBB";
        }

        string GetPrefix() const
        {
            return "bruteforce_parallel_tbb";
        }

        unique_ptr<ResultT> Process(ProblemT& problem) override
        {
            int numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->template CreateNeighborsContainer<NeighborsContainerT>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            typedef blocked_range<PointVectorIteratorT> point_range_t;

            task_scheduler_init scheduler(task_scheduler_init::deferred);

            if (numThreads > 0)
            {
                scheduler.initialize(numThreads);
            }
            else
            {
                scheduler.initialize(task_scheduler_init::automatic);
            }

            auto start = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            parallel_for(point_range_t(inputDatasetBegin, inputDatasetEnd), [&](point_range_t& range)
                {
                    auto rangeBegin = range.begin();
                    auto rangeEnd = range.end();

                    for (auto inputPoint = rangeBegin; inputPoint < rangeEnd; ++inputPoint)
                    {
                        auto& neighbors = pNeighborsContainer->at(inputPoint->id - 1);

                        for (auto trainingPoint = trainingDatasetBegin; trainingPoint < trainingDatasetEnd; ++trainingPoint)
                        {
                            this->AddNeighbor(inputPoint, trainingPoint, neighbors);
                        }
                    }
                }
            );

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<ResultT>(new ResultT(problem, GetPrefix(), pNeighborsContainer, elapsed, chrono::duration<double>()));
        }

    private:
        int numThreads = 0;

};

#endif // BRUTEFORCEPARALLELTBBALGORITHM_H
