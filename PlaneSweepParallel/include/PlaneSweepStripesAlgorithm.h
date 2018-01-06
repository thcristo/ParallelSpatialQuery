#ifndef PLANESWEEPSTRIPESALGORITHM_H
#define PLANESWEEPSTRIPESALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include "AllKnnResultStripes.h"

template<class ProblemT, class ResultT, class ResultBaseT, class PointVectorT, class PointVectorIteratorT, class NeighborsContainerT, class StripeDataT>
class PlaneSweepStripesAlgorithm : public AbstractAllKnnAlgorithm<ProblemT, ResultBaseT, PointVectorT, PointVectorIteratorT>
{
    public:
        PlaneSweepStripesAlgorithm(int numStripes) : numStripes(numStripes)
        {
        }

        virtual ~PlaneSweepStripesAlgorithm() {}

        string GetTitle() const
        {
            return "Plane sweep stripes";
        }

        string GetPrefix() const
        {
            return "planesweep_stripes";
        }

        unique_ptr<ResultBaseT> Process(ProblemT& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->template CreateNeighborsContainer<NeighborsContainerT>(problem.GetInputDataset(), numNeighbors);

            int numStripesLocal = omp_get_max_threads();

            if (numStripes > 0)
            {
                numStripesLocal = numStripes;
            }

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<ResultT>(new ResultT(problem, GetPrefix()));

            auto stripeData = pResult->GetStripeData(numStripesLocal);
            numStripesLocal = stripeData.InputDatasetStripe.size();

            auto finishSorting = chrono::high_resolution_clock::now();

            for (int iStripeInput = 0; iStripeInput < numStripesLocal; ++iStripeInput)
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
                    bool highStripeEnd = iStripeTrainingNext >= numStripesLocal;

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
                                highStripeEnd = iStripeTrainingNext >= numStripesLocal;
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

        void PlaneSweepStripe(PointVectorIteratorT inputPointIter, StripeDataT stripeData, int iStripeTraining,
                              PointNeighbors<neighbors_priority_queue_t>& neighbors, double mindy) const
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
                    if (this->CheckAddNeighbor(inputPointIter, prevTrainingPointIter, neighbors, mindy))
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
                    if (this->CheckAddNeighbor(inputPointIter, nextTrainingPointIter, neighbors, mindy))
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

#endif // PLANESWEEPSTRIPESALGORITHM_H
