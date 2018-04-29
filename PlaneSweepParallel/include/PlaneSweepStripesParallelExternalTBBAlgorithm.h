#ifndef PLANESWEEPSTRIPESPARALLELEXTERNALTBBALGORITHM_H
#define PLANESWEEPSTRIPESPARALLELEXTERNALTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <tbb/tbb.h>

class PlaneSweepStripesParallelExternalTBBAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepStripesParallelExternalTBBAlgorithm(size_t numStripes, int numThreads, bool parallelSort, bool splitByT) : numStripes(numStripes),
            numThreads(numThreads), parallelSort(parallelSort), splitByT(splitByT)
        {
        }

        virtual ~PlaneSweepStripesParallelExternalTBBAlgorithm() {}

        bool UsesExternalMemory() override
        {
            return true;
        }

        string GetTitle() const
        {
            if (splitByT)
                return parallelSort ? "Plane sweep stripes parallel external TBB (parallel sorting, split by training)" : "Plane sweep stripes parallel external TBB (split by training)";
            else
                return parallelSort ? "Plane sweep stripes parallel external TBB (parallel sorting)" : "Plane sweep stripes parallel external TBB";
        }

        string GetPrefix() const
        {
            if (splitByT)
                return parallelSort ? "planesweep_stripes_parallel_external_TBB_psort_splitByT" : "planesweep_stripes_parallel_external_TBB_splitByT";
            else
                return parallelSort ? "planesweep_stripes_parallel_external_TBB_psort" : "planesweep_stripes_parallel_external_TBB";
        }
    protected:

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            int numThreadsToUse = omp_get_max_threads();

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
                numThreadsToUse = numThreads;
            }

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<AllKnnResultStripesParallelExternal>(
                                new AllKnnResultStripesParallelExternal(static_cast<AllKnnProblemExternal&>(problem),
                                                GetPrefix(), parallelSort, splitByT));

            numStripes = pResult->SplitStripes(numStripes);

            auto finishSorting = chrono::high_resolution_clock::now();

            size_t startStripe = 0;
            size_t endStripe = 0;
            size_t nextStripe = 0;

            do
            {
                auto pWindow = pResult->GetWindow(nextStripe, false);
                startStripe = pWindow->GetStartStripe();
                endStripe = pWindow->GetEndStripe();
                nextStripe = endStripe + 1;

                if (pWindow != nullptr)
                    PlaneSweepWindow(pWindow, pResult, numThreadsToUse);
                else
                    break;

            } while (endStripe < numStripes - 1);

            if (!pResult->HasAllocationError())
            {
                bool hasPrevStripe = startStripe > 0;

                while (hasPrevStripe)
                {
                    auto pWindow = pResult->GetWindow(startStripe-1, true);
                    startStripe = pWindow->GetStartStripe();

                    hasPrevStripe = startStripe > 0;

                    if (pWindow != nullptr)
                        PlaneSweepWindow(pWindow, pResult, numThreadsToUse);
                    else
                        break;
                }
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            pResult->setDuration(elapsed);
            pResult->setDurationSorting(elapsedSorting);

            return pResult;
        }

    private:
        size_t numStripes = 0;
        int numThreads = 0;
        bool parallelSort = false;
        bool splitByT = false;

        void PlaneSweepWindow(unique_ptr<StripesWindow>& pWindow, unique_ptr<AllKnnResultStripesParallelExternal>& pResult, unsigned int numThreadsToUse)
        {
            bool isSecondPass = pWindow->IsSecondPass();
            size_t windowStartStripe = pWindow->GetStartStripe();
            size_t windowEndStripe = pWindow->GetEndStripe();
            size_t numWindowStripes = pWindow->GetNumStripes();
            auto stripeData = pWindow->GetStripeData();

            auto pPendingPointsContainer = pResult->GetPendingPointsForWindow(*pWindow);
            auto pendingPointsIterBegin = pPendingPointsContainer->cbegin();
            auto pendingPointsIterEnd = pPendingPointsContainer->cend();

            auto& pendingNeighborsContainer = pResult->GetPendingNeighborsContainer();

            typedef blocked_range<point_vector_t::const_iterator> point_range_t;

            tbb::parallel_for(point_range_t(pendingPointsIterBegin, pendingPointsIterEnd), [&](point_range_t& range)
                {
                    auto rangeBegin = range.begin();
                    auto rangeEnd = range.end();

                    for (auto inputPointIter = rangeBegin; inputPointIter < rangeEnd; ++inputPointIter)
                    {
                        bool exit = false;
                        auto& neighbors = pendingNeighborsContainer.at(inputPointIter->id);
                        size_t lowStripe = neighbors.getLowStripe();
                        size_t highStripe = neighbors.getHighStripe();
                        size_t currentStripe = isSecondPass ? lowStripe - 1 : highStripe + 1;
                        int step = isSecondPass ? -1 : 1;

                        do
                        {
                            double dy = isSecondPass ? inputPointIter->y - stripeData.StripeBoundaries[currentStripe - windowStartStripe].maxY
                                        : stripeData.StripeBoundaries[currentStripe - windowStartStripe].minY - inputPointIter->y;
                            double dySquared = dy*dy;
                            if (dySquared < neighbors.MaxDistanceElement().distanceSquared)
                            {
                                PlaneSweepStripe(inputPointIter, stripeData, currentStripe - windowStartStripe, neighbors, dySquared);
                                exit = isSecondPass ? currentStripe == windowStartStripe : currentStripe == windowEndStripe;

                                if (exit)
                                {
                                    if (isSecondPass)
                                        neighbors.setLowStripe(currentStripe);
                                    else
                                        neighbors.setHighStripe(currentStripe);
                                }
                            }
                            else
                            {
                                exit = true;
                                if (isSecondPass)
                                    neighbors.setLowStripe(0);
                                else
                                    neighbors.setHighStripe(numStripes - 1);
                            }

                            if (!exit)
                            {
                                currentStripe += step;
                            }
                        } while (!exit);
                    }
                }
            );

            if (!isSecondPass)
            {
                auto& neighborsContainer = pWindow->GetNeighborsContainer();

                if (numWindowStripes >= numThreadsToUse)
                {
                    tbb::parallel_for(blocked_range<size_t>(windowStartStripe, windowEndStripe+1), [&](blocked_range<size_t>& range)
                        {
                            auto rangeBegin = range.begin();
                            auto rangeEnd = range.end();

                            for (size_t iStripeInput = rangeBegin; iStripeInput < rangeEnd; ++iStripeInput)
                            {
                                auto& inputDataset = stripeData.InputDatasetStripe[iStripeInput - windowStartStripe];
                                auto inputDatasetBegin = inputDataset.cbegin();
                                auto inputDatasetEnd = inputDataset.cend();
                                auto& stripeNeighborsContainer = neighborsContainer[iStripeInput - windowStartStripe];

                                for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
                                {
                                    size_t iStripeTraining = iStripeInput;
                                    size_t inputPointOffset = std::distance(inputDatasetBegin, inputPointIter);
                                    auto& neighbors = stripeNeighborsContainer.at(inputPointOffset);

                                    PlaneSweepStripe(inputPointIter, stripeData, iStripeTraining - windowStartStripe, neighbors, 0.0);

                                    bool lowStripeEnd = iStripeTraining == windowStartStripe;
                                    bool highStripeEnd = iStripeTraining == windowEndStripe;

                                    size_t iStripeTrainingPrev = lowStripeEnd ? iStripeTraining : iStripeTraining - 1;
                                    size_t iStripeTrainingNext = highStripeEnd ? iStripeTraining : iStripeTraining + 1;

                                    if (lowStripeEnd)
                                    {
                                        neighbors.setLowStripe(iStripeTraining);
                                    }

                                    if (highStripeEnd)
                                    {
                                        neighbors.setHighStripe(iStripeTraining);
                                    }

                                    while (!lowStripeEnd || !highStripeEnd)
                                    {
                                        if (!lowStripeEnd)
                                        {
                                            double dyLow = inputPointIter->y - stripeData.StripeBoundaries[iStripeTrainingPrev - windowStartStripe].maxY;
                                            double dySquaredLow = dyLow*dyLow;
                                            if (dySquaredLow < neighbors.MaxDistanceElement().distanceSquared)
                                            {
                                                PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingPrev - windowStartStripe, neighbors, dySquaredLow);
                                                lowStripeEnd = iStripeTrainingPrev == windowStartStripe;

                                                if (lowStripeEnd)
                                                    neighbors.setLowStripe(windowStartStripe);
                                                else
                                                    --iStripeTrainingPrev;
                                            }
                                            else
                                            {
                                                lowStripeEnd = true;
                                                neighbors.setLowStripe(0);
                                            }
                                        }

                                        if (!highStripeEnd)
                                        {
                                            double dyHigh = stripeData.StripeBoundaries[iStripeTrainingNext - windowStartStripe].minY - inputPointIter->y;
                                            double dySquaredHigh = dyHigh*dyHigh;
                                            if (dySquaredHigh < neighbors.MaxDistanceElement().distanceSquared)
                                            {
                                                PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingNext - windowStartStripe, neighbors, dySquaredHigh);
                                                highStripeEnd = iStripeTrainingNext == windowEndStripe;

                                                if (highStripeEnd)
                                                    neighbors.setHighStripe(windowEndStripe);
                                                else
                                                    ++iStripeTrainingNext;
                                            }
                                            else
                                            {
                                                highStripeEnd = true;
                                                neighbors.setHighStripe(numStripes - 1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    );
                }
                else
                {
                    for (size_t iStripeInput = windowStartStripe; iStripeInput <= windowEndStripe; ++iStripeInput)
                    {
                        auto& inputDataset = stripeData.InputDatasetStripe[iStripeInput - windowStartStripe];
                        auto inputDatasetBegin = inputDataset.cbegin();
                        auto inputDatasetEnd = inputDataset.cend();
                        auto& stripeNeighborsContainer = neighborsContainer[iStripeInput - windowStartStripe];

                        tbb::parallel_for(point_range_t(inputDatasetBegin, inputDatasetEnd), [&](point_range_t& range)
                            {
                                auto rangeBegin = range.begin();
                                auto rangeEnd = range.end();

                                for (auto inputPointIter = rangeBegin; inputPointIter < rangeEnd; ++inputPointIter)
                                {
                                    size_t iStripeTraining = iStripeInput;
                                    size_t inputPointOffset = std::distance(inputDatasetBegin, inputPointIter);
                                    auto& neighbors = stripeNeighborsContainer.at(inputPointOffset);

                                    PlaneSweepStripe(inputPointIter, stripeData, iStripeTraining - windowStartStripe, neighbors, 0.0);

                                    bool lowStripeEnd = iStripeTraining == windowStartStripe;
                                    bool highStripeEnd = iStripeTraining == windowEndStripe;

                                    size_t iStripeTrainingPrev = lowStripeEnd ? iStripeTraining : iStripeTraining - 1;
                                    size_t iStripeTrainingNext = highStripeEnd ? iStripeTraining : iStripeTraining + 1;

                                    if (lowStripeEnd)
                                    {
                                        neighbors.setLowStripe(iStripeTraining);
                                    }

                                    if (highStripeEnd)
                                    {
                                        neighbors.setHighStripe(iStripeTraining);
                                    }

                                    while (!lowStripeEnd || !highStripeEnd)
                                    {
                                        if (!lowStripeEnd)
                                        {
                                            double dyLow = inputPointIter->y - stripeData.StripeBoundaries[iStripeTrainingPrev - windowStartStripe].maxY;
                                            double dySquaredLow = dyLow*dyLow;
                                            if (dySquaredLow < neighbors.MaxDistanceElement().distanceSquared)
                                            {
                                                PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingPrev - windowStartStripe, neighbors, dySquaredLow);
                                                lowStripeEnd = iStripeTrainingPrev == windowStartStripe;

                                                if (lowStripeEnd)
                                                    neighbors.setLowStripe(windowStartStripe);
                                                else
                                                    --iStripeTrainingPrev;
                                            }
                                            else
                                            {
                                                lowStripeEnd = true;
                                                neighbors.setLowStripe(0);
                                            }
                                        }

                                        if (!highStripeEnd)
                                        {
                                            double dyHigh = stripeData.StripeBoundaries[iStripeTrainingNext - windowStartStripe].minY - inputPointIter->y;
                                            double dySquaredHigh = dyHigh*dyHigh;
                                            if (dySquaredHigh < neighbors.MaxDistanceElement().distanceSquared)
                                            {
                                                PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingNext - windowStartStripe, neighbors, dySquaredHigh);
                                                highStripeEnd = iStripeTrainingNext == windowEndStripe;

                                                if (highStripeEnd)
                                                    neighbors.setHighStripe(windowEndStripe);
                                                else
                                                    ++iStripeTrainingNext;
                                            }
                                            else
                                            {
                                                highStripeEnd = true;
                                                neighbors.setHighStripe(numStripes - 1);
                                            }
                                        }
                                    }
                                }
                            }
                        );
                    }
                }
            }

            pResult->CommitWindow(*pWindow, *pPendingPointsContainer);
        }

        void PlaneSweepStripe(point_vector_iterator_t inputPointIter, StripeData stripeData, size_t iStripeTraining,
                              PointNeighbors<neighbors_priority_queue_t>& neighbors, double mindy) const
        {
            auto& trainingDataset = stripeData.TrainingDatasetStripe[iStripeTraining];

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();

            if (trainingDatasetBegin == trainingDatasetEnd)
                return;

            auto nextTrainingPointIter = lower_bound(trainingDatasetBegin, trainingDatasetEnd, inputPointIter->x,
                        [](const Point& point, const double& value) { return point.x < value; } );

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
                    if (CheckAddNeighbor(inputPointIter, prevTrainingPointIter, neighbors, mindy))
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
                    if (CheckAddNeighbor(inputPointIter, nextTrainingPointIter, neighbors, mindy))
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
};

#endif // PLANESWEEPSTRIPESPARALLELEXTERNALTBBALGORITHM_H
