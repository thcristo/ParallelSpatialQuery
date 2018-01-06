#ifndef PLANESWEEPCOPYPARALLELTBBALGORITHM_H
#define PLANESWEEPCOPYPARALLELTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <tbb/tbb.h>

using namespace tbb;

template<class ProblemT, class ResultT, class ResultBaseT, class PointVectorT, class PointVectorIteratorT, class NeighborsContainerT>
class PlaneSweepCopyParallelTBBAlgorithm : public AbstractAllKnnAlgorithm<ProblemT, ResultBaseT, PointVectorT, PointVectorIteratorT, NeighborsContainerT>
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

        unique_ptr<ResultBaseT> Process(ProblemT& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

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

            auto pResult = unique_ptr<ResultT>(new ResultT(problem, GetPrefix(), parallelSort));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDataset = pResult->GetTrainingDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            parallel_for(point_range_t(inputDatasetBegin, inputDatasetEnd), [&](point_range_t& range)
                {
                    auto rangeBegin = range.begin();
                    auto rangeEnd = range.end();

                    /*
                    auto startSearchPos = lower_bound(trainingDatasetBegin, trainingDatasetEnd, rangeBegin->x,
                                            [&](const Point& point, const double& value) { return point.x < value; } );
                    */
                    for (auto inputPointIter = rangeBegin; inputPointIter < rangeEnd; ++inputPointIter)
                    {
                        auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);


                        auto nextTrainingPointIter = lower_bound(trainingDatasetBegin, trainingDatasetEnd, inputPointIter->x,
                                            [&](const Point& point, const double& value) { return point.x < value; } );


                        /*
                        auto nextTrainingPointIter = startSearchPos;
                        while (nextTrainingPointIter < trainingDatasetEnd && nextTrainingPointIter->x < inputPointIter->x)
                        {
                            ++nextTrainingPointIter;
                        }

                        startSearchPos = nextTrainingPointIter;
                        */

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
                                if (this->CheckAddNeighbor(inputPointIter, prevTrainingPointIter, neighbors))
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
                                if (this->CheckAddNeighbor(inputPointIter, nextTrainingPointIter, neighbors))
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
        int numThreads = 0;
        bool parallelSort = false;

};

#endif // PLANESWEEPCOPYPARALLELTBBALGORITHM_H
