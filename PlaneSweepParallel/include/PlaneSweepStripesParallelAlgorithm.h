#ifndef PLANESWEEPSTRIPESPARALLELALGORITHM_H
#define PLANESWEEPSTRIPESPARALLELALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include "AllKnnResultStripesParallel.h"

class PlaneSweepStripesParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepStripesParallelAlgorithm(int numStripes, int numThreads, bool parallelSort, bool splitByT) : numStripes(numStripes),
            numThreads(numThreads), parallelSort(parallelSort), splitByT(splitByT)
        {
        }

        virtual ~PlaneSweepStripesParallelAlgorithm() {}

        string GetTitle() const
        {
            if (splitByT)
                return parallelSort ? "Plane sweep stripes parallel (parallel sorting, split by training)" : "Plane sweep stripes parallel (split by training)";
            else
                return parallelSort ? "Plane sweep stripes parallel (parallel sorting)" : "Plane sweep stripes parallel";
        }

        string GetPrefix() const
        {
            if (splitByT)
                return parallelSort ? "planesweep_stripes_parallel_psort_splitByT" : "planesweep_stripes_parallel_splitByT";
            else
                return parallelSort ? "planesweep_stripes_parallel_psort" : "planesweep_stripes_parallel";
        }

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
            }

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<AllKnnResultStripesParallel>(new AllKnnResultStripesParallel(problem, GetPrefix(), parallelSort, splitByT));

            auto stripeData = pResult->GetStripeData(numStripes);

            numStripes = stripeData.InputDatasetStripe.size();

            auto finishSorting = chrono::high_resolution_clock::now();

            #pragma omp parallel for schedule(dynamic)
            for (int iStripeInput = 0; iStripeInput < numStripes; ++iStripeInput)
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
        int numStripes = 0;
        int numThreads = 0;
        bool parallelSort = false;
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

#endif // PLANESWEEPSTRIPESPARALLELALGORITHM_H
