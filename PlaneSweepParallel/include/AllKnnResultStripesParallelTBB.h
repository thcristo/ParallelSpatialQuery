/* This file contains a class definition of AkNN result for striped plane sweep algorithm.
    The splitting process is implemented with a parallel for loop instead of a serial while loop
    The implementation is exactly the same as AllKnnResultStripesParallel with the only difference of using Intel TBB
    instead of OpenMP for parallel operations
 */
#ifndef ALLKNNRESULTSTRIPESPARALLELTBB_H
#define ALLKNNRESULTSTRIPESPARALLELTBB_H

#include "AllKnnResultStripes.h"

/** \brief Class definition of AkNN result of striped plane sweep algorithm (TBB version)
 */
class AllKnnResultStripesParallelTBB : public AllKnnResultStripes
{
    public:
        AllKnnResultStripesParallelTBB(const AllKnnProblem& problem, const string& filePrefix) : AllKnnResultStripes(problem, filePrefix)
        {
        }

        AllKnnResultStripesParallelTBB(const AllKnnProblem& problem, const string& filePrefix, bool parallelSort, bool splitByT) : AllKnnResultStripes(problem, filePrefix, parallelSort, splitByT)
        {
        }
        virtual ~AllKnnResultStripesParallelTBB() {}

    protected:

        void create_fixed_stripes_input(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY) override
        {
            size_t inputDatasetStripeSize = inputDatasetSortedY.size()/numStripes;
            auto inputDatasetSortedYBegin = inputDatasetSortedY.cbegin();
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();
            auto trainingDatasetSortedYBegin = trainingDatasetSortedY.cbegin();
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();

            size_t numRemainingPoints = inputDatasetSortedY.size() % numStripes;
            if (numRemainingPoints != 0)
            {
                numStripes += (numRemainingPoints/inputDatasetStripeSize + 1);
            }

            pInputDatasetStripe->resize(numStripes, point_vector_t());
            pTrainingDatasetStripe->resize(numStripes, point_vector_t());
            pStripeBoundaries->resize(numStripes, {0.0, 0.0});

            parallel_for(blocked_range<size_t>(0, numStripes), [&](blocked_range<size_t>& range)
            {
                auto rangeBegin = range.begin();
                auto rangeEnd = range.end();

                for (size_t i=rangeBegin; i < rangeEnd; ++i)
                {
                    StripeBoundaries_t& stripeBoundaries = pStripeBoundaries->at(i);
                    point_vector_t& inputStripe = pInputDatasetStripe->at(i);
                    point_vector_t& trainingStripe = pTrainingDatasetStripe->at(i);

                    auto inputIterStart = inputDatasetSortedYBegin + i*inputDatasetStripeSize;
                    auto inputIterEnd = inputIterStart;
                    if ((size_t)distance(inputIterStart, inputDatasetSortedYEnd) <= inputDatasetStripeSize)
                        inputIterEnd = inputDatasetSortedYEnd;
                    else
                    {
                        inputIterEnd = inputIterStart + inputDatasetStripeSize;
                        auto inputIterEndLimit = inputDatasetSortedYEnd;

                        if ((size_t)distance(inputIterEnd, inputDatasetSortedYEnd) > inputDatasetStripeSize)
                            inputIterEndLimit = inputIterEnd + inputDatasetStripeSize;

                        while (inputIterEnd < inputIterEndLimit && (prev(inputIterEnd))->y == inputIterEnd->y)
                            ++inputIterEnd;
                    }

                    if (i > 0)
                    {
                        while (inputIterStart < inputIterEnd && (prev(inputIterStart))->y == inputIterStart->y)
                            ++inputIterStart;
                    }

                    if (inputIterStart < inputIterEnd)
                    {
                        inputStripe.assign(inputIterStart, inputIterEnd);
                        sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
                             {
                                 return point1.x < point2.x;
                             });

                        stripeBoundaries.minY =  i > 0 ? inputIterStart->y : 0.0;
                        stripeBoundaries.maxY =  i < numStripes - 1 ? (inputIterEnd < inputDatasetSortedYEnd ? inputIterEnd->y : 1.0001) : 1.0001;

                        auto trainingIterStart = lower_bound(trainingDatasetSortedYBegin, trainingDatasetSortedYEnd, stripeBoundaries.minY,
                                                        [](const Point& point, const double& value) { return point.y < value; });

                        auto trainingIterEnd = trainingIterStart;
                        while (trainingIterEnd < trainingDatasetSortedYEnd && trainingIterEnd->y < stripeBoundaries.maxY)
                            ++trainingIterEnd;

                        if (trainingIterStart < trainingIterEnd)
                        {
                            trainingStripe.assign(trainingIterStart, trainingIterEnd);
                            sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
                             {
                                 return point1.x < point2.x;
                             });
                        }
                    }
                    else
                    {
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
            });
        }

        void create_fixed_stripes_training(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY) override
        {
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

            parallel_for(blocked_range<size_t>(0, numStripes), [&](blocked_range<size_t>& range)
            {
                auto rangeBegin = range.begin();
                auto rangeEnd = range.end();

                for (size_t i=rangeBegin; i < rangeEnd; ++i)
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
            });
        }
};

#endif // ALLKNNRESULTSTRIPESPARALLELTBB_H
