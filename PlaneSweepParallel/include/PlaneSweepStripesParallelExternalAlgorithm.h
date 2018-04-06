#ifndef PLANESWEEPSTRIPESPARALLELEXTERNALALGORITHM_H
#define PLANESWEEPSTRIPESPARALLELEXTERNALALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include "StripesWindow.h"
#include "AllKnnResultStripesParallelExternal.h"

class PlaneSweepStripesParallelExternalAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepStripesParallelExternalAlgorithm(int numStripes, int numThreads, bool parallelSort, bool splitByT) : numStripes(numStripes),
            numThreads(numThreads), parallelSort(parallelSort), splitByT(splitByT)
        {
        }
        virtual ~PlaneSweepStripesParallelExternalAlgorithm() {}

        string GetTitle() const
        {
            if (splitByT)
                return parallelSort ? "Plane sweep stripes parallel external (parallel sorting, split by training)" : "Plane sweep stripes parallel external (split by training)";
            else
                return parallelSort ? "Plane sweep stripes parallel external (parallel sorting)" : "Plane sweep stripes parallel external";
        }

        string GetPrefix() const
        {
            if (splitByT)
                return parallelSort ? "planesweep_stripes_parallel_external_psort_splitByT" : "planesweep_stripes_parallel_external_splitByT";
            else
                return parallelSort ? "planesweep_stripes_parallel_external_psort" : "planesweep_stripes_parallel_external";
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

            auto pResult = unique_ptr<AllKnnResultStripesParallelExternal>(new AllKnnResultStripesParallelExternal(problem, GetPrefix(), parallelSort, splitByT));

            numStripes = pResult->SplitStripes(numStripes);

            auto finishSorting = chrono::high_resolution_clock::now();

            int startStripe = -1;
            int endStripe = -1;

            do
            {
                auto pWindow = pResult->GetWindow(endStripe + 1, false);
                startStripe = pWindow->GetStartStripe();
                endStripe = pWindow->GetEndStripe();

                if (pWindow != nullptr)
                    PlaneSweepWindow(pWindow, pResult, numThreadsToUse);
                else
                    break;

            } while (endStripe < numStripes - 1);

            if (!pResult->HasAllocationError())
            {
                while (startStripe > 0)
                {
                    auto pWindow = pResult->GetWindow(startStripe - 1, true);
                    startStripe = pWindow->GetStartStripe();
                    endStripe = pWindow->GetEndStripe();

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
        int numStripes = 0;
        int numThreads = 0;
        bool parallelSort = false;
        bool splitByT = false;

        void PlaneSweepWindow(unique_ptr<StripesWindow>& pWindow, unique_ptr<AllKnnResultStripesParallelExternal>& pResult, int numThreadsToUse)
        {
            bool isSecondPass = pWindow->IsSecondPass();
            int windowStartStripe = pWindow->GetStartStripe();
            int windowEndStripe = pWindow->GetEndStripe();
            int numWindowStripes = pWindow->GetNumStripes();
            auto stripeData = pWindow->GetStripeData();

            auto pPendingPointsContainer = pResult->GetPendingPointsForWindow(*pWindow);
            auto pendingPointsIterBegin = pPendingPointsContainer->cbegin();
            auto pendingPointsIterEnd = pPendingPointsContainer->cend();

            auto& pendingNeighborsContainer = pResult->GetPendingNeighborsContainer();

            #pragma omp parallel for schedule(dynamic, 10)
            for (auto inputPointIter = pendingPointsIterBegin; inputPointIter < pendingPointsIterEnd; ++inputPointIter)
            {
                bool exit = false;
                int currentStripe = isSecondPass ? windowEndStripe : windowStartStripe;
                int step = isSecondPass ? -1 : 1;
                auto& neighbors = pendingNeighborsContainer.at(inputPointIter->id);

                do
                {
                    double dy = isSecondPass ? inputPointIter->y - stripeData.StripeBoundaries[currentStripe].maxY
                                : stripeData.StripeBoundaries[currentStripe].minY - inputPointIter->y;
                    double dySquared = dy*dy;
                    if (dySquared < neighbors.MaxDistanceElement().distanceSquared)
                    {
                        PlaneSweepStripe(inputPointIter, stripeData, currentStripe, neighbors, dySquared);
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
                        currentStripe += step;
                } while (!exit);
            }

            if (!isSecondPass)
            {
                auto& neighborsContainer = pWindow->GetNeighborsContainer();

                #pragma omp parallel for schedule(dynamic) if (numWindowStripes >= numThreadsToUse)
                for (int iStripeInput = windowStartStripe; iStripeInput <= windowEndStripe; ++iStripeInput)
                {
                    auto& inputDataset = stripeData.InputDatasetStripe[iStripeInput];
                    auto inputDatasetBegin = inputDataset.cbegin();
                    auto inputDatasetEnd = inputDataset.cend();

                    #pragma omp parallel for if (numWindowStripes < numThreadsToUse)
                    for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
                    {
                        int iStripeTraining = iStripeInput;
                        auto& neighbors = neighborsContainer.at(inputPointIter->id);

                        PlaneSweepStripe(inputPointIter, stripeData, iStripeTraining, neighbors, 0.0);

                        int iStripeTrainingPrev = iStripeTraining - 1;
                        int iStripeTrainingNext = iStripeTraining + 1;
                        bool lowStripeEnd = iStripeTrainingPrev < windowStartStripe;
                        bool highStripeEnd = iStripeTrainingNext > windowEndStripe;

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
                                double dyLow = inputPointIter->y - stripeData.StripeBoundaries[iStripeTrainingPrev].maxY;
                                double dySquaredLow = dyLow*dyLow;
                                if (dySquaredLow < neighbors.MaxDistanceElement().distanceSquared)
                                {
                                    PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingPrev, neighbors, dySquaredLow);
                                    --iStripeTrainingPrev;
                                    lowStripeEnd = iStripeTrainingPrev < windowStartStripe;
                                    if (lowStripeEnd)
                                        neighbors.setLowStripe(windowStartStripe);
                                }
                                else
                                {
                                    lowStripeEnd = true;
                                    neighbors.setLowStripe(0);
                                }
                            }

                            if (!highStripeEnd)
                            {
                                double dyHigh = stripeData.StripeBoundaries[iStripeTrainingNext].minY - inputPointIter->y;
                                double dySquaredHigh = dyHigh*dyHigh;
                                if (dySquaredHigh < neighbors.MaxDistanceElement().distanceSquared)
                                {
                                    PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingNext, neighbors, dySquaredHigh);
                                    ++iStripeTrainingNext;
                                    highStripeEnd = iStripeTrainingNext > windowEndStripe;
                                    if (highStripeEnd)
                                        neighbors.setHighStripe(windowEndStripe);
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

            pWindow->CommitWindow();
        }

        void PlaneSweepStripe(point_vector_iterator_t inputPointIter, StripeData stripeData, int iStripeTraining,
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

#endif // PLANESWEEPSTRIPESPARALLELEXTERNALALGORITHM_H
