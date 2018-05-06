#ifndef PLANESWEEPSTRIPESPARALLELTBBALGORITHM_H
#define PLANESWEEPSTRIPESPARALLELTBBALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include "AllKnnResultStripesParallelTBB.h"

class PlaneSweepStripesParallelTBBAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepStripesParallelTBBAlgorithm(int numStripes, int numThreads, bool parallelSort,bool parallelSplit, bool splitByT) : numStripes(numStripes),
            numThreads(numThreads), parallelSort(parallelSort), parallelSplit(parallelSplit), splitByT(splitByT)
        {
        }

        virtual ~PlaneSweepStripesParallelTBBAlgorithm() {}

        string GetTitle() const
        {
            stringstream ss;

            ss << "Plane sweep stripes parallel TBB, parallelSort=" << parallelSort << ", parallelSplit=" << parallelSplit << ", splitByTraining=" << splitByT;
            return ss.str();
        }

        string GetPrefix() const
        {
            stringstream ss;

            ss << "planesweep_stripes_parallel_TBB_psort_" << parallelSort << "_psplit_" << parallelSplit << "_splitByT_" << splitByT;
            return ss.str();
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

            unique_ptr<AllKnnResultStripes> pResult;

            if (parallelSplit)
                pResult.reset(new AllKnnResultStripesParallelTBB(problem, GetPrefix(), parallelSort, splitByT));
            else
                pResult.reset(new AllKnnResultStripes(problem, GetPrefix(), parallelSort, splitByT));

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

                            PlaneSweepStripe(inputPointIter, stripeData, iStripeTraining, neighbors, 0.0);

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
                                        PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingPrev, neighbors, dySquaredLow);
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
                                        PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingNext, neighbors, dySquaredHigh);
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

    private:
        int numStripes = 0;
        int numThreads = 0;
        bool parallelSort = false;
        bool parallelSplit = false;
        bool splitByT = false;

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

#endif // PLANESWEEPSTRIPESPARALLELTBBALGORITHM_H
