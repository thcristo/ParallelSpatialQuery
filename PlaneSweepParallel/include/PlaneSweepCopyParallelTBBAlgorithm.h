/* Parallel plane sweep algorithm implementation using Intel TBB
    This implementation operates on a copy of the original datasets and it is based on PlaneSweepCopyAlgorithm
 */
#ifndef PLANESWEEPCOPYPARALLELTBBALGORITHM_H
#define PLANESWEEPCOPYPARALLELTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <tbb/tbb.h>

using namespace tbb;

/** \brief Parallel plane sweep algorithm using copy of original datasets and Intel TBB
 */
class PlaneSweepCopyParallelTBBAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepCopyParallelTBBAlgorithm(int numThreads, bool parallelSort) : numThreads(numThreads), parallelSort(parallelSort)
        {
        }

        virtual ~PlaneSweepCopyParallelTBBAlgorithm() {}

        string GetTitle() const
        {
            return parallelSort ? "Plane sweep copy parallel TBB (parallel sorting)" : "Plane sweep copy parallel TBB";
        }

        string GetPrefix() const
        {
            return parallelSort ? "planesweep_copy_parallel_TBB_psort" : "planesweep_copy_parallel_TBB";
        }

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            //the implementation is similar to PlaneSweepCopyAlgorithm
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

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

            auto pResult = unique_ptr<AllKnnResultSorted>(new AllKnnResultSorted(problem, GetPrefix(), parallelSort));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDataset = pResult->GetTrainingDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            //Intel TBB parallel loop, the input dataset is recursively split into ranges and each range is assigned to a thread
            parallel_for(point_range_t(inputDatasetBegin, inputDatasetEnd), [&](point_range_t& range)
                {
                    auto rangeBegin = range.begin();
                    auto rangeEnd = range.end();

                    //loop through input points of this range
                    for (auto inputPointIter = rangeBegin; inputPointIter < rangeEnd; ++inputPointIter)
                    {
                        auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);


                        auto nextTrainingPointIter = lower_bound(trainingDatasetBegin, trainingDatasetEnd, inputPointIter->x,
                                            [&](const Point& point, const double& value) { return point.x < value; } );

                        auto prevTrainingPointIter = nextTrainingPointIter;
                        if (prevTrainingPointIter > trainingDatasetBegin)
                        {
                            --prevTrainingPointIter;
                        }

                        bool lowStop = prevTrainingPointIter == nextTrainingPointIter;
                        bool highStop = nextTrainingPointIter == trainingDatasetEnd;

                        //start moving left and right of the input point in x axis
                        while (!lowStop || !highStop)
                        {
                            if (!lowStop)
                            {
                                if (CheckAddNeighbor(inputPointIter, prevTrainingPointIter, neighbors))
                                {
                                    if (prevTrainingPointIter > trainingDatasetBegin)
                                    {
                                        --prevTrainingPointIter;
                                    }
                                    else
                                    {
                                        lowStop = true;
                                    }
                                }
                                else
                                {
                                    lowStop = true;
                                }
                            }

                            if (!highStop)
                            {
                                if (CheckAddNeighbor(inputPointIter, nextTrainingPointIter, neighbors))
                                {
                                    if (nextTrainingPointIter < trainingDatasetEnd)
                                    {
                                        ++nextTrainingPointIter;
                                    }

                                    if (nextTrainingPointIter == trainingDatasetEnd)
                                    {
                                        highStop = true;
                                    }
                                }
                                else
                                {
                                    highStop = true;
                                }
                            }
                        }
                    }
                }
            );

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            pResult->setDuration(elapsed);
            pResult->setDurationSorting(elapsedSorting);
            pResult->setNeighborsContainer(pNeighborsContainer);

            return pResult;
        }

    private:
        int numThreads = 0;
        bool parallelSort = false;

};

#endif // PLANESWEEPCOPYPARALLELTBBALGORITHM_H
