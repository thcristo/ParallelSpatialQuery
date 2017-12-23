#ifndef PLANESWEEPSTRIPESPARALLELTBBALGORITHM_H
#define PLANESWEEPSTRIPESPARALLELTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"


class PlaneSweepStripesParallelTBBAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepStripesParallelTBBAlgorithm(int numStripes, int numThreads, bool parallelSort) : numStripes(numStripes),
            numThreads(numThreads), parallelSort(parallelSort)
        {
        }

        virtual ~PlaneSweepStripesParallelTBBAlgorithm() {}

        string GetTitle() const
        {
            return parallelSort ? "Plane sweep stripes parallel TBB (parallel sorting)" : "Plane sweep stripes parallel TBB";
        }

        string GetPrefix() const
        {
            return parallelSort ? "planesweep_stripes_parallel_TBB_psort" : "planesweep_stripes_parallel_TBB";
        }

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

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

            auto pResult = unique_ptr<AllKnnResultStripes>(new AllKnnResultStripes(problem, GetPrefix(), parallelSort));

            auto stripeData = pResult->GetStripeData(numStripes);

            numStripes = stripeData.InputDatasetStripe.size();

            auto finishSorting = chrono::high_resolution_clock::now();

            parallel_for(blocked_range<int>(0, numStripes), [&](blocked_range<int>& range)
                {
                    auto rangeBegin = range.begin();
                    auto rangeEnd = range.end();

                    for (int iStripeInput = rangeBegin; iStripeInput < rangeEnd; ++iStripeInput)
                    {
                        auto& inputDataset = stripeData.InputDatasetStripe[iStripeInput];
                        auto inputDatasetBegin = inputDataset.cbegin();
                        auto inputDatasetEnd = inputDataset.cend();

                        for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
                        {
                            int iStripeTraining = iStripeInput;
                            auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                            PlaneSweepStripe(inputPointIter, stripeData, iStripeTraining, neighbors);

                            int iStripeTrainingPrev = iStripeTraining - 1;
                            int iStripeTrainingNext = iStripeTraining + 1;
                            bool lowStripeEnd = iStripeTrainingPrev < 0;
                            bool highStripeEnd = iStripeTrainingNext >= numStripes;

                            while (!lowStripeEnd || !highStripeEnd)
                            {
                                if (!lowStripeEnd)
                                {
                                    double dyLow = inputPointIter->y - stripeData.StripeBoundaries[iStripeTrainingPrev].maxY;
                                    double dySquaredLow = dyLow*dyLow;
                                    if (dySquaredLow < neighbors.MaxDistanceElement().distanceSquared)
                                    {
                                        PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingPrev, neighbors);
                                        --iStripeTrainingPrev;
                                        lowStripeEnd = iStripeTrainingPrev < 0;
                                    }
                                    else
                                    {
                                        lowStripeEnd = true;
                                    }
                                }

                                if (!highStripeEnd)
                                {
                                    double dyHigh = stripeData.StripeBoundaries[iStripeTrainingNext].minY - inputPointIter->y;
                                    double dySquaredHigh = dyHigh*dyHigh;
                                    if (dySquaredHigh < neighbors.MaxDistanceElement().distanceSquared)
                                    {
                                        PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingNext, neighbors);
                                        ++iStripeTrainingNext;
                                        highStripeEnd = iStripeTrainingNext >= numStripes;
                                    }
                                    else
                                    {
                                        highStripeEnd = true;
                                    }
                                }
                            }
                        }
                    }
                });

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
        int numStripes;
        int numThreads;
        bool parallelSort;

        void PlaneSweepStripe(point_vector_iterator_t inputPointIter, StripeData stripeData, int iStripeTraining,
                              PointNeighbors<neighbors_priority_queue_t>& neighbors) const
        {
            auto& trainingDataset = stripeData.TrainingDatasetStripe[iStripeTraining];

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();

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
};

#endif // PLANESWEEPSTRIPESPARALLELTBBALGORITHM_H