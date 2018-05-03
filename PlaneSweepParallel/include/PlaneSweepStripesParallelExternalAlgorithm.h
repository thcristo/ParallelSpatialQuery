#ifndef PLANESWEEPSTRIPESPARALLELEXTERNALALGORITHM_H
#define PLANESWEEPSTRIPESPARALLELEXTERNALALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include "StripesWindow.h"
#include "AllKnnResultStripesParallelExternal.h"

class PlaneSweepStripesParallelExternalAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepStripesParallelExternalAlgorithm(size_t numStripes, int numThreads, bool parallelSort, bool splitByT) : numStripes(numStripes),
            numThreads(numThreads), parallelSort(parallelSort), splitByT(splitByT)
        {
        }
        virtual ~PlaneSweepStripesParallelExternalAlgorithm() {}

        bool UsesExternalMemory() override
        {
            return true;
        }

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

            auto pResult = unique_ptr<AllKnnResultStripesParallelExternal>(
                                new AllKnnResultStripesParallelExternal(static_cast<AllKnnProblemExternal&>(problem),
                                                GetPrefix(), parallelSort, splitByT));

            cout << "split stripes start" << endl;
            numStripes = pResult->SplitStripes(numStripes);
            cout << "split stripes end" << endl;
            auto finishSorting = chrono::high_resolution_clock::now();

            size_t startStripe = 0;
            size_t endStripe = 0;
            size_t nextStripe = 0;

            cout << "first pass started" << endl;

            do
            {
                auto pWindow = pResult->GetWindow(nextStripe, false);

                if (pWindow != nullptr)
                {
                    startStripe = pWindow->GetStartStripe();
                    endStripe = pWindow->GetEndStripe();
                    nextStripe = endStripe + 1;
                    cout << "got window " << startStripe << " " << endStripe << endl;
                    PlaneSweepWindow(pWindow, pResult, numThreadsToUse);
                    cout << "processed window" << endl;
                }
                else
                    break;

            } while (endStripe < numStripes - 1);

            cout << "first pass ended" << endl;

            if (!pResult->HasAllocationError())
            {
                cout << "second pass started" << endl;

                bool hasPrevStripe = startStripe > 0;

                while (hasPrevStripe)
                {
                    auto pWindow = pResult->GetWindow(startStripe-1, true);

                    if (pWindow != nullptr)
                    {
                        startStripe = pWindow->GetStartStripe();
                        endStripe = pWindow->GetEndStripe();
                        hasPrevStripe = startStripe > 0;
                        cout << "got window " << startStripe << " " << endStripe << endl;
                        PlaneSweepWindow(pWindow, pResult, numThreadsToUse);
                        cout << "processed window" << endl;
                    }
                    else
                        break;
                }

                cout << "second pass ended" << endl;

                cout << "neighbors sort start" << endl;
                pResult->SortNeighbors();
                cout << "neighbors sort end" << endl;
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            pResult->setDuration(elapsed);
            pResult->setDurationSorting(elapsedSorting);
            pResult->CalcHeapStats();

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

            #pragma omp parallel for schedule(dynamic, 10)
            for (auto inputPointIter = pendingPointsIterBegin; inputPointIter < pendingPointsIterEnd; ++inputPointIter)
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

            if (!isSecondPass)
            {
                auto& neighborsContainer = pWindow->GetNeighborsContainer();

                #pragma omp parallel for schedule(dynamic) if (numWindowStripes >= numThreadsToUse)
                for (size_t iStripeInput = windowStartStripe; iStripeInput <= windowEndStripe; ++iStripeInput)
                {
                    auto& inputDataset = stripeData.InputDatasetStripe[iStripeInput - windowStartStripe];
                    auto inputDatasetBegin = inputDataset.cbegin();
                    auto inputDatasetEnd = inputDataset.cend();
                    auto& stripeNeighborsContainer = neighborsContainer[iStripeInput - windowStartStripe];

                    #pragma omp parallel for if (numWindowStripes < numThreadsToUse)
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

            cout << "commit window started" << endl;
            pResult->CommitWindow(*pWindow, *pPendingPointsContainer);
            cout << "commit window ended" << endl;
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

#endif // PLANESWEEPSTRIPESPARALLELEXTERNALALGORITHM_H
