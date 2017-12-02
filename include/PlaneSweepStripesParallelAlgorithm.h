#ifndef PLANESWEEPSTRIPESPARALLELALGORITHM_H
#define PLANESWEEPSTRIPESPARALLELALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>


class PlaneSweepStripesParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepStripesParallelAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }
        virtual ~PlaneSweepStripesParallelAlgorithm() {}

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            int numStripes = omp_get_max_threads();
            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
                numStripes = numThreads;
            }

            auto start = chrono::high_resolution_clock::now();

            auto stripeData = problem.GetStripeData(numStripes);

            auto finishSorting = chrono::high_resolution_clock::now();

            #pragma omp parallel for
            for (int iStripeInput = 0; iStripeInput < numStripes; ++iStripeInput)
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

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, elapsedSorting, "planesweep_stripes_parallel", problem));
        }
    protected:

    private:
        int numThreads;

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

#endif // PLANESWEEPSTRIPESPARALLELALGORITHM_H
