#ifndef ALLKNNRESULTSTRIPESPARALLEL_H
#define ALLKNNRESULTSTRIPESPARALLEL_H

#include "AllKnnResultStripes.h"


class AllKnnResultStripesParallel : public AllKnnResultStripes
{
    public:
        AllKnnResultStripesParallel(const AllKnnProblem& problem, const string& filePrefix) : AllKnnResultStripes(problem, filePrefix)
        {
        }

        AllKnnResultStripesParallel(const AllKnnProblem& problem, const string& filePrefix, bool parallelSort, bool splitByT) : AllKnnResultStripes(problem, filePrefix, parallelSort), splitByT(splitByT)
        {
        }
        virtual ~AllKnnResultStripesParallel() {}

    protected:
        void create_fixed_stripes(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY) override
        {
            if (splitByT)
                create_fixed_stripes_training(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            else
                create_fixed_stripes_input(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
        }

    private:
        bool splitByT = false;

        void create_fixed_stripes_input(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY)
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

            #pragma omp parallel for schedule(dynamic)
            for (size_t i=0; i < numStripes; ++i)
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

                    auto trainingIterEnd = lower_bound(trainingDatasetSortedYBegin, trainingDatasetSortedYEnd, stripeBoundaries.maxY,
                                                        [](const Point& point, const double& value) { return point.y < value; });

                    auto trainingIterStart = lower_bound(trainingDatasetSortedYBegin, trainingIterEnd, stripeBoundaries.minY,
                                                        [](const Point& point, const double& value) { return point.y < value; });

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
        }

        void create_fixed_stripes_training(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY)
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

                    auto inputIterEnd = lower_bound(inputDatasetSortedYBegin, inputDatasetSortedYEnd, stripeBoundaries.maxY,
                                                        [](const Point& point, const double& value) { return point.y < value; });

                    auto inputIterStart = lower_bound(inputDatasetSortedYBegin, inputIterEnd, stripeBoundaries.minY,
                                                        [](const Point& point, const double& value) { return point.y < value; });

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

            /*
            for (size_t i=0; i < numStripes; ++i)
            {
                point_vector_t& inputVector = pInputDatasetStripe->at(i);
                size_t numInputPoints = inputVector.size();
                if (numInputPoints > 0)
                {
                    pStripeIndex->insert(pStripeIndex->end(), numInputPoints, i);
                    pInputPoints->insert(pInputPoints->end(), inputVector.begin(), inputVector.end());
                }
            }
            */
        }
};

#endif // ALLKNNRESULTSTRIPESPARALLEL_H
