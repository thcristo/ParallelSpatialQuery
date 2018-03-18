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
            size_t numNeighbors = problem.GetNumNeighbors();

            /*
            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);
            */

            int numThreadsToUse = omp_get_max_threads();

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
                numThreadsToUse = numThreads;
            }

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<AllKnnResultStripesParallelExternal>(new AllKnnResultStripesParallelExternal(problem, GetPrefix(), parallelSort, splitByT));

            numStripes = pResult->SplitStripes();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto pWindow = pResult->GetMainWindow();
            if (pWindow != nullptr)
                PlaneSweepWindow(pWindow, pResult);
            else
                throw ApplicationException("Cannot allocate memory for main window of stripes.");

            int lowStartStripe = pWindow->GetStartStripe();
            int highEndStripe = pWindow->GetEndStripe();

            bool lowWindowEnd = (lowStartStripe == 0);
            bool highWindowEnd = (highEndStripe == numStripes - 1);

            while (!lowWindowEnd || !highWindowEnd)
            {
                if (!lowWindowEnd)
                {
                    pWindow = pResult->GetLowWindow(lowStartStripe);
                    if (pWindow != nullptr)
                        PlaneSweepWindow(pWindow, pResult, numThreadsToUse);
                    else
                        throw ApplicationException("Cannot allocate memory for low window of stripes.");
                    lowStartStripe = pWindow->GetStartStripe();
                    lowWindowEnd = (lowStartStripe == 0);
                }

                if (!highWindowEnd)
                {
                    pWindow = pResult->GetHighWindow(highEndStripe);
                    if (pWindow != nullptr)
                        PlaneSweepWindow(pWindow, pResult, numThreadsToUse);
                    else
                        throw ApplicationException("Cannot allocate memory for high window of stripes.");
                    highEndStripe = pWindow->GetEndStripe();
                    highWindowEnd = (highEndStripe == numStripes - 1);
                }

            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            pResult->setDuration(elapsed);
            pResult->setDurationSorting(elapsedSorting);
            pResult->setNeighborsContainer(pNeighborsContainer);

            return pResult;
        }

    private:
        int numStripes = 0;
        int numThreads = 0;
        bool parallelSort = false;
        bool splitByT = false;

        void PlaneSweepWindow(unique_ptr<StripesWindow>& pWindow, unique_ptr<AllKnnResultStripesParallelExternal>& pResult, int numThreadsToUse)
        {
            auto pNeighborsContainer = pWindow->GetNeighborsContainer();
            int windowStartStripe = pWindow->GetStartStripe();
            int windowEndStripe = pWindow->GetEndStripe();
            int numWindowStripes = pWindow->GetNumStripes();
            auto& stripeData = pWindow->GetStripeData();
            bool isLow = pWindow->IsLow();
            bool isHigh = pWindow->IsHigh();

            if (isLow || isHigh)
            {
                auto pendingPointsIter = pResult->GetPendingPoints();
                auto pendingPointsIterBegin = pendingPointsIter.cbegin();
                auto pendingPointsIterEnd = pendingPointsIter.cend();

                auto pPendingNeighborsContainer = pResult->GetPendingNeighborsContainer();

                #pragma omp parallel for schedule(dynamic, 10)
                for (auto inputPointIter = pendingPointsIterBegin; inputPointIter < pendingPointsIterEnd; ++inputPointIter)
                {
                    bool exit = false;
                    int currentStripe = isLow ? windowEndStripe : windowStartStripe;
                    int step = isLow ? -1 : 1;
                    auto& neighbors = pPendingNeighborsContainer->at(inputPointIter->id);

                    do
                    {
                        double dy = isLow ? inputPointIter->y - stripeData.StripeBoundaries[currentStripe].maxY
                                    : stripeData.StripeBoundaries[currentStripe].minY - inputPointIter->y;
                        double dySquared = dy*dy;
                        if (dySquared < neighbors.MaxDistanceElement().distanceSquared)
                        {
                            PlaneSweepStripe(inputPointIter, stripeData, currentStripe, neighbors, dySquared);
                            exit = isLow ? currentStripe == windowStartStripe : currentStripe == windowEndStripe;

                            if (exit)
                            {
                                if (isLow)
                                    neighbors.setLowStripe(currentStripe);
                                else
                                    neighbors.setHighStripe(currentStripe);
                            }
                        }
                        else
                        {
                            exit = true;
                            if (isLow)
                                    neighbors.setLowStripe(0);
                                else
                                    neighbors.setHighStripe(numStripes - 1);
                        }

                        if (!exit)
                            currentStripe += step;


                    } while (!exit)
                }
            }

            #pragma omp parallel for schedule(dynamic) if (numWindowStripes >= numThreadsToUse)
            for (int iStripeInput = windowStartStripe; iStripeInput <= windowEndStripe; ++iStripeInput)
            {
                auto& inputDataset = pWindow->GetInputDatasetStripe(iStripeInput);
                auto inputDatasetBegin = inputDataset.cbegin();
                auto inputDatasetEnd = inputDataset.cend();

                #pragma omp parallel for if (numWindowStripes < numThreadsToUse)
                for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
                {
                    int iStripeTraining = iStripeInput;
                    auto& neighbors = pNeighborsContainer->at(inputPointIter->id);

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

                    pResult->CheckAddPendingPoint(inputPointIter, neighbors);
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
                /*
                if (!lowStop && !highStop)
                {
                    auto continuations = CheckAddNeighbors(inputPointIter, prevTrainingPointIter, nextTrainingPointIter, neighbors);
                    if (continuations[0])
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

                    if (continuations[1])
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
                */

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
