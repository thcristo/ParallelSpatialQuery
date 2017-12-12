#ifndef PLANESWEEPCOPYPARALLELTBBALGORITHM_H
#define PLANESWEEPCOPYPARALLELTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <tbb/tbb.h>

using namespace tbb;


class PlaneSweepCopyParallelTBBAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepCopyParallelTBBAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }

        virtual ~PlaneSweepCopyParallelTBBAlgorithm() {}

        string GetTitle() const
        {
            return "Plane sweep copy parallel TBB";
        }

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            typedef blocked_range<point_vector_t::const_iterator> point_range_t;

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<AllKnnResultSorted>(new AllKnnResultSorted(problem, "planesweep_copy_parallel_TBB"));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDataset = pResult->GetTrainingDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

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

                    auto startSearchPos = lower_bound(trainingDatasetBegin, trainingDatasetEnd, rangeBegin->x,
                                            [&](const Point& point, const double& value) { return point.x < value; } );

                    for (auto inputPointIter = rangeBegin; inputPointIter < rangeEnd; ++inputPointIter)
                    {
                        auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                        /*
                        auto nextTrainingPointIter = lower_bound(startSearchPos, trainingDataset.cend(), inputPointIter->x,
                                            [&](const Point& point, const double& value) { return point.x < value; } );
                        */

                        auto nextTrainingPointIter = startSearchPos;
                        while (nextTrainingPointIter < trainingDatasetEnd && nextTrainingPointIter->x < inputPointIter->x)
                        {
                            ++nextTrainingPointIter;
                        }

                        startSearchPos = nextTrainingPointIter;
                        auto prevTrainingPointIter = nextTrainingPointIter;
                        if (prevTrainingPointIter > trainingDatasetBegin)
                        {
                            --prevTrainingPointIter;
                        }

                        bool lowStop = prevTrainingPointIter == nextTrainingPointIter;
                        bool highStop = nextTrainingPointIter == trainingDatasetEnd;

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

    protected:

    private:
        int numThreads;

};

#endif // PLANESWEEPCOPYPARALLELTBBALGORITHM_H
