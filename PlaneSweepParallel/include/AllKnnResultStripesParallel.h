/* This file contains a class definition of AkNN result for striped plane sweep algorithm.
    The splitting process is implemented with a parallel for loop instead of a serial while loop
    OpenMP is used for parallel operations
 */

#ifndef ALLKNNRESULTSTRIPESPARALLEL_H
#define ALLKNNRESULTSTRIPESPARALLEL_H

#include "AllKnnResultStripes.h"


/** \brief Class definition of AkNN result of striped plane sweep algorithm
 */
class AllKnnResultStripesParallel : public AllKnnResultStripes
{
    public:
        AllKnnResultStripesParallel(const AllKnnProblem& problem, const std::string& filePrefix) : AllKnnResultStripes(problem, filePrefix)
        {
        }

        AllKnnResultStripesParallel(const AllKnnProblem& problem, const std::string& filePrefix, bool parallelSort, bool splitByT) : AllKnnResultStripes(problem, filePrefix, parallelSort, splitByT)
        {
        }
        virtual ~AllKnnResultStripesParallel() {}

    protected:

        /** \brief Splits the datasets into stripes based on the input dataset (fixed number of input points per stripe)
         *
         * \param numStripes size_t the desired number of stripes
         * \param inputDatasetSortedY const point_vector_t& the sorted input dataset
         * \param trainingDatasetSortedY const point_vector_t& the sorted training dataset
         * \return void
         *
         */
        void create_fixed_stripes_input(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY) override
        {
            //calculate the number of input points per stripe based on the number of stripes
            size_t inputDatasetStripeSize = inputDatasetSortedY.size()/numStripes;
            auto inputDatasetSortedYBegin = inputDatasetSortedY.cbegin();
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();
            auto trainingDatasetSortedYBegin = trainingDatasetSortedY.cbegin();
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();

            size_t numRemainingPoints = inputDatasetSortedY.size() % numStripes;
            if (numRemainingPoints != 0)
            {
                //increase the number of stripes to include the remaining input points
                numStripes += (numRemainingPoints/inputDatasetStripeSize + 1);
            }

            //vectors for stripes are allocated
            pInputDatasetStripe->resize(numStripes, point_vector_t());
            pTrainingDatasetStripe->resize(numStripes, point_vector_t());
            pStripeBoundaries->resize(numStripes, {0.0, 0.0});

            //this loop is executed in parallel, each repetition creates one stripe
            #pragma omp parallel for schedule(dynamic)
            for (size_t i=0; i < numStripes; ++i)
            {
                //get the vector to use for storing the points
                StripeBoundaries_t& stripeBoundaries = pStripeBoundaries->at(i);
                point_vector_t& inputStripe = pInputDatasetStripe->at(i);
                point_vector_t& trainingStripe = pTrainingDatasetStripe->at(i);

                //find the beginning of the current stripe
                auto inputIterStart = inputDatasetSortedYBegin + i*inputDatasetStripeSize;
                auto inputIterEnd = inputIterStart;
                if ((size_t)distance(inputIterStart, inputDatasetSortedYEnd) <= inputDatasetStripeSize)
                    inputIterEnd = inputDatasetSortedYEnd; //in this case we are in the last stripe
                else
                {
                    inputIterEnd = inputIterStart + inputDatasetStripeSize;
                    auto inputIterEndLimit = inputDatasetSortedYEnd;

                    //find the end of current stripe
                    if ((size_t)distance(inputIterEnd, inputDatasetSortedYEnd) > inputDatasetStripeSize)
                        inputIterEndLimit = inputIterEnd + inputDatasetStripeSize;

                    //advance the end of current stripe to include any input points having same y
                    while (inputIterEnd < inputIterEndLimit && (prev(inputIterEnd))->y == inputIterEnd->y)
                        ++inputIterEnd;
                }

                if (i > 0)
                {
                    //advance the beginning of current stripe to the next different y
                    while (inputIterStart < inputIterEnd && (prev(inputIterStart))->y == inputIterStart->y)
                        ++inputIterStart;
                }

                if (inputIterStart < inputIterEnd)
                {
                    //we found the input points for current stripe
                    inputStripe.assign(inputIterStart, inputIterEnd);
                    //sort the input points by using serial sort
                    sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });

                    //find the boundaries of current stripe
                    stripeBoundaries.minY =  i > 0 ? inputIterStart->y : 0.0;
                    stripeBoundaries.maxY =  i < numStripes - 1 ? (inputIterEnd < inputDatasetSortedYEnd ? inputIterEnd->y : 1.0001) : 1.0001;

                    //do a binary search to find the first training point of current stripe
                    //unfortunately we cannot use the end point of previous stripe because the loop runs in parallel execution
                    auto trainingIterStart = lower_bound(trainingDatasetSortedYBegin, trainingDatasetSortedYEnd, stripeBoundaries.minY,
                                                        [](const Point& point, const double& value) { return point.y < value; });

                    auto trainingIterEnd = trainingIterStart;
                    //advance the end of current stripe to include training points with same y
                    while (trainingIterEnd < trainingDatasetSortedYEnd && trainingIterEnd->y < stripeBoundaries.maxY)
                        ++trainingIterEnd;

                    if (trainingIterStart < trainingIterEnd)
                    {
                        //we found training points for current stripe
                        trainingStripe.assign(trainingIterStart, trainingIterEnd);
                        //sort training points by x using a serial sort
                        sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                    }
                }
                else
                {
                    //in this case we have an empty stripe, it will be ignored by the algorithm
                    if (inputIterStart >= inputDatasetSortedYEnd)
                    {
                        stripeBoundaries.minY = 1.0001;
                        stripeBoundaries.maxY = 1.0001;
                    }
                    else
                    {
                        stripeBoundaries.minY = inputIterStart->y;
                        stripeBoundaries.maxY = inputIterStart->y;
                    }
                }
            }
        }

        /** \brief Splits the datasets into stripes based on the training dataset (fixed number of training points per stripe)
         *
         * \param numStripes size_t the desired number of stripes
         * \param inputDatasetSortedY const point_vector_t& the sorted input dataset
         * \param trainingDatasetSortedY const point_vector_t& the sorted training dataset
         * \return void
         *
         */
        void create_fixed_stripes_training(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY) override
        {
            //The implementation is exactly the same as create_fixed_stripes_input
            //with the only difference of switching between input and training datasets
            size_t trainingDatasetStripeSize = trainingDatasetSortedY.size()/numStripes;
            auto inputDatasetSortedYBegin = inputDatasetSortedY.cbegin();
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();
            auto trainingDatasetSortedYBegin = trainingDatasetSortedY.cbegin();
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();

            size_t numRemainingPoints = trainingDatasetSortedY.size() % numStripes;
            if (numRemainingPoints != 0)
            {
                numStripes += (numRemainingPoints/trainingDatasetStripeSize + 1);
            }

            pInputDatasetStripe->resize(numStripes, point_vector_t());
            pTrainingDatasetStripe->resize(numStripes, point_vector_t());
            pStripeBoundaries->resize(numStripes, {0.0, 0.0});

            #pragma omp parallel for schedule(dynamic)
            for (size_t i=0; i < numStripes; ++i)
            {
                StripeBoundaries_t& stripeBoundaries = pStripeBoundaries->at(i);
                point_vector_t& inputStripe = pInputDatasetStripe->at(i);
                point_vector_t& trainingStripe = pTrainingDatasetStripe->at(i);

                auto trainingIterStart = trainingDatasetSortedYBegin + i*trainingDatasetStripeSize;
                auto trainingIterEnd = trainingIterStart;
                if ((size_t)distance(trainingIterStart, trainingDatasetSortedYEnd) <= trainingDatasetStripeSize)
                    trainingIterEnd = trainingDatasetSortedYEnd;
                else
                {
                    trainingIterEnd = trainingIterStart + trainingDatasetStripeSize;
                    auto trainingIterEndLimit = trainingDatasetSortedYEnd;

                    if ((size_t)distance(trainingIterEnd, trainingDatasetSortedYEnd) > trainingDatasetStripeSize)
                        trainingIterEndLimit = trainingIterEnd + trainingDatasetStripeSize;

                    while (trainingIterEnd < trainingIterEndLimit && (prev(trainingIterEnd))->y == trainingIterEnd->y)
                        ++trainingIterEnd;
                }

                if (i > 0)
                {
                    while (trainingIterStart < trainingIterEnd && (prev(trainingIterStart))->y == trainingIterStart->y)
                        ++trainingIterStart;
                }

                if (trainingIterStart < trainingIterEnd)
                {
                    trainingStripe.assign(trainingIterStart, trainingIterEnd);
                    sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });

                    stripeBoundaries.minY =  i > 0 ? trainingIterStart->y : 0.0;
                    stripeBoundaries.maxY =  i < numStripes - 1 ? (trainingIterEnd < trainingDatasetSortedYEnd ? trainingIterEnd->y : 1.0001) : 1.0001;

                    auto inputIterStart = lower_bound(inputDatasetSortedYBegin, inputDatasetSortedYEnd, stripeBoundaries.minY,
                                                        [](const Point& point, const double& value) { return point.y < value; });

                    auto inputIterEnd = inputIterStart;
                    while (inputIterEnd < inputDatasetSortedYEnd && inputIterEnd->y < stripeBoundaries.maxY)
                        ++inputIterEnd;

                    if (inputIterStart < inputIterEnd)
                    {
                        inputStripe.assign(inputIterStart, inputIterEnd);
                        sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                    }
                }
                else
                {
                    if (trainingIterStart >= trainingDatasetSortedYEnd)
                    {
                        stripeBoundaries.minY = 1.0001;
                        stripeBoundaries.maxY = 1.0001;
                    }
                    else
                    {
                        stripeBoundaries.minY = trainingIterStart->y;
                        stripeBoundaries.maxY = trainingIterStart->y;
                    }
                }
            }
        }
};

#endif // ALLKNNRESULTSTRIPESPARALLEL_H
