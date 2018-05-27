/* Serial plane sweep algorithm with stripes  */
#ifndef PLANESWEEPSTRIPESALGORITHM_H
#define PLANESWEEPSTRIPESALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include "AllKnnResultStripes.h"

/** \brief Serial plane sweep with stripes
 */
class PlaneSweepStripesAlgorithm : public AbstractAllKnnAlgorithm
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

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            //allocate vector of max heaps to store neighbors
            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            int numStripesLocal = omp_get_max_threads();

            //check if a user-defined number of stripes was requested, otherwise use the number of cores as number of stripes
            if (numStripes > 0)
            {
                numStripesLocal = numStripes;
            }

            auto start = chrono::high_resolution_clock::now();

            //create result object
            auto pResult = unique_ptr<AllKnnResultStripes>(new AllKnnResultStripes(problem, GetPrefix()));

            //split datasets into stripes
            auto stripeData = pResult->GetStripeData(numStripesLocal);
            //get the actual number of stripes
            numStripesLocal = stripeData.InputDatasetStripe.size();
            //record the time used for splitting stripes
            auto finishSorting = chrono::high_resolution_clock::now();

            //serial loop through all input points
            for (int iStripeInput = 0; iStripeInput < numStripesLocal; ++iStripeInput)
            {
                auto& inputDataset = stripeData.InputDatasetStripe[iStripeInput];
                auto inputDatasetBegin = inputDataset.cbegin();
                auto inputDatasetEnd = inputDataset.cend();

                //loop through all input points of current stripe
                for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
                {
                    int iStripeTraining = iStripeInput;
                    auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                    //first check for neighbors in the same stripe
                    PlaneSweepStripe(inputPointIter, stripeData, iStripeTraining, neighbors, 0.0);

                    int iStripeTrainingPrev = iStripeTraining - 1;
                    int iStripeTrainingNext = iStripeTraining + 1;
                    bool lowStripeEnd = iStripeTrainingPrev < 0;
                    bool highStripeEnd = iStripeTrainingNext >= numStripesLocal;

                    //now check for neighbors in other stripes moving alternately to higher and lower y
                    while (!lowStripeEnd || !highStripeEnd)
                    {
                        //look for lower y
                        if (!lowStripeEnd)
                        {
                            //check the distance of input point from the high boundary of the stripe
                            double dyLow = inputPointIter->y - stripeData.StripeBoundaries[iStripeTrainingPrev].maxY;
                            double dySquaredLow = dyLow*dyLow;
                            if (dySquaredLow < neighbors.MaxDistanceElement().distanceSquared)
                            {
                                //examine points in this stripe
                                PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingPrev, neighbors, dySquaredLow);
                                --iStripeTrainingPrev;
                                lowStripeEnd = iStripeTrainingPrev < 0;
                            }
                            else
                            {
                                //distance from boundary is greater than top of the heap, stop looking to lower stripes
                                lowStripeEnd = true;
                            }
                        }

                        //look for higher y
                        if (!highStripeEnd)
                        {
                            //check the distance of input point from the low boundary of the stripe
                            double dyHigh = stripeData.StripeBoundaries[iStripeTrainingNext].minY - inputPointIter->y;
                            double dySquaredHigh = dyHigh*dyHigh;
                            if (dySquaredHigh < neighbors.MaxDistanceElement().distanceSquared)
                            {
                                //examine points in this stripe
                                PlaneSweepStripe(inputPointIter, stripeData, iStripeTrainingNext, neighbors, dySquaredHigh);
                                ++iStripeTrainingNext;
                                highStripeEnd = iStripeTrainingNext >= numStripesLocal;
                            }
                            else
                            {
                                //distance from boundary is greater than top of the heap, stop looking to lower stripes
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

    private:
        int numStripes = 0;

        /** \brief Searches for neighbors of an input point in a specific stripe
         *
         * \param inputPointIter point_vector_iterator_t iterator pointing to input point
         * \param stripeData StripeData data for all stripes
         * \param iStripeTraining int index of stripe to be examined
         * \param neighbors PointNeighbors<neighbors_priority_queue_t>& object containing the max heap of neighbors for the given input point
         * \param mindy double squared distance of input point from the nearest boundary of the stripe
         * \return
         *
         */
        void PlaneSweepStripe(point_vector_iterator_t inputPointIter, StripeData stripeData, int iStripeTraining,
                              PointNeighbors<neighbors_priority_queue_t>& neighbors, double mindy) const
        {
            auto& trainingDataset = stripeData.TrainingDatasetStripe[iStripeTraining];

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();

            //do a binary search to find the next training point in x axis
            auto nextTrainingPointIter = lower_bound(trainingDatasetBegin, trainingDatasetEnd, inputPointIter->x,
                        [](const Point& point, const double& value) { return point.x < value; } );

            //find the previous training point
            auto prevTrainingPointIter = nextTrainingPointIter;
            if (prevTrainingPointIter > trainingDatasetBegin)
            {
                --prevTrainingPointIter;
            }

            bool lowStop = prevTrainingPointIter == nextTrainingPointIter;
            bool highStop = nextTrainingPointIter == trainingDatasetEnd;

            //start moving to left and right of input point
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

#endif // PLANESWEEPSTRIPESALGORITHM_H
