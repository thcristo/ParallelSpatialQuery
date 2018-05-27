/* Parallel plane sweep algorithm with stripes (OpenMP implementation) */

#ifndef PLANESWEEPSTRIPESPARALLELALGORITHM_H
#define PLANESWEEPSTRIPESPARALLELALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include "AllKnnResultStripesParallel.h"

/** \brief Parallel plane sweep with stripes (OpenMP)
 */
class PlaneSweepStripesParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        /** \brief Constructor
         *
         * \param numStripes int number of stripes to use
         * \param numThreads int number of threads to use
         * \param parallelSort bool true if we want to use parallel sorting of stripe points by x
         * \param parallelSplit bool true if we want to use parallel splitting of datasets to stripes
         * \param splitByT bool true if we want to split stripes by using the training dataset
         */
        PlaneSweepStripesParallelAlgorithm(int numStripes, int numThreads, bool parallelSort, bool parallelSplit, bool splitByT) : numStripes(numStripes),
            numThreads(numThreads), parallelSort(parallelSort), parallelSplit(parallelSplit), splitByT(splitByT)
        {
        }

        virtual ~PlaneSweepStripesParallelAlgorithm() {}

        string GetTitle() const
        {
            stringstream ss;

            ss << "Plane sweep stripes parallel, parallelSort=" << parallelSort << ", parallelSplit=" << parallelSplit << ", splitByTraining=" << splitByT;
            return ss.str();
        }

        string GetPrefix() const
        {
            stringstream ss;

            ss << "planesweep_stripes_parallel_psort_" << parallelSort << "_psplit_" << parallelSplit << "_splitByT_" << splitByT;
            return ss.str();
        }

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            //the implementation is similar to PlaneSweepStripesAlgorithm
            size_t numNeighbors = problem.GetNumNeighbors();

            //allocate vector of neighbors for all input points
            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            //if numThreads=0, let the system decide the number of threads based on number of cores
            if (numThreads > 0)
            {
                //set the number of threads explicitly
                omp_set_num_threads(numThreads);
            }

            auto start = chrono::high_resolution_clock::now();

            unique_ptr<AllKnnResultStripes> pResult;

            //we use a different class for result depending on splitting method
            if (parallelSplit)
                pResult.reset(new AllKnnResultStripesParallel(problem, GetPrefix(), parallelSort, splitByT));
            else
                pResult.reset(new AllKnnResultStripes(problem, GetPrefix(), parallelSort, splitByT));

            //split datasets into stripes
            auto stripeData = pResult->GetStripeData(numStripes);

            //get the actual number of stripes (may be slightly more than the desired number)
            numStripes = stripeData.InputDatasetStripe.size();

            auto finishSorting = chrono::high_resolution_clock::now();

            //parallel loop through all stripes
            //we use dynamic scheduling so thread scheduling is based on the workload of each stripe
            #pragma omp parallel for schedule(dynamic)
            for (int iStripeInput = 0; iStripeInput < numStripes; ++iStripeInput)
            {
                auto& inputDataset = stripeData.InputDatasetStripe[iStripeInput];
                auto inputDatasetBegin = inputDataset.cbegin();
                auto inputDatasetEnd = inputDataset.cend();

                //loop through all points of current stripe
                for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
                {
                    int iStripeTraining = iStripeInput;
                    auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                    //first check for neighbors in the same stripe
                    PlaneSweepStripe(inputPointIter, stripeData, iStripeTraining, neighbors, 0.0);

                    int iStripeTrainingPrev = iStripeTraining - 1;
                    int iStripeTrainingNext = iStripeTraining + 1;
                    bool lowStripeEnd = iStripeTrainingPrev < 0;
                    bool highStripeEnd = iStripeTrainingNext >= numStripes;

                    //now check for neighbors in other stripes moving alternately to higher and lower y
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
        bool parallelSplit = false;
        bool splitByT = false;

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
            //the implementation is the same as PlaneSweepStripesAlgorithm
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
