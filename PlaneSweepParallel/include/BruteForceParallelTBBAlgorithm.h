/* Parallel brute force algorithm implementation using Intel TBB */

#ifndef BRUTEFORCEPARALLELTBBALGORITHM_H
#define BRUTEFORCEPARALLELTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <tbb/tbb.h>

using namespace tbb;

/** \brief Parallel brute force algorithm using Intel TBB
 */
class BruteForceParallelTBBAlgorithm : public AbstractAllKnnAlgorithm
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

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            int numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            typedef blocked_range<point_vector_t::const_iterator> point_range_t;

            task_scheduler_init scheduler(task_scheduler_init::deferred);

            if (numThreads > 0)
            {
                //user-defined number of threads
                scheduler.initialize(numThreads);
            }
            else
            {
                //automatic number of threads equal to the number of cores
                scheduler.initialize(task_scheduler_init::automatic);
            }

            auto start = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            //Intel TBB parallel loop, the input dataset is recursively split into ranges and each range is assigned to a thread
            parallel_for(point_range_t(inputDatasetBegin, inputDatasetEnd), [&](point_range_t& range)
                {
                    //get the beginning and end of current range
                    auto rangeBegin = range.begin();
                    auto rangeEnd = range.end();

                    //loop through input points of this range
                    for (auto inputPoint = rangeBegin; inputPoint < rangeEnd; ++inputPoint)
                    {
                        auto& neighbors = pNeighborsContainer->at(inputPoint->id - 1);

                        //loop through all training points
                        for (auto trainingPoint = trainingDatasetBegin; trainingPoint < trainingDatasetEnd; ++trainingPoint)
                        {
                            AddNeighbor(inputPoint, trainingPoint, neighbors);
                        }
                    }
                }
            );

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(problem, GetPrefix(), pNeighborsContainer, elapsed, chrono::duration<double>()));
        }

    private:
        int numThreads = 0;

};

#endif // BRUTEFORCEPARALLELTBBALGORITHM_H
