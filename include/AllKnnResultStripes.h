#ifndef ALLKNNRESULTSTRIPES_H
#define ALLKNNRESULTSTRIPES_H

#include "AllKnnResult.h"


class AllKnnResultStripes : public AllKnnResult
{
    public:
        AllKnnResultStripes(const AllKnnProblem& problem, const string& filePrefix) : AllKnnResult(problem, filePrefix)
        {
        }

        virtual ~AllKnnResultStripes() {}

        StripeData GetStripeData(int numStripes)
        {
            if (!pInputDatasetStripe)
            {
                pInputDatasetStripe.reset(new point_vector_vector_t());
            }

            if (!pTrainingDatasetStripe)
            {
                pTrainingDatasetStripe.reset(new point_vector_vector_t());
            }

            if (!pStripeBoundaries)
            {
                pStripeBoundaries.reset(new vector<StripeBoundaries>());
            }

            point_vector_t inputDatasetSortedY(problem.GetInputDataset());
            point_vector_t trainingDatasetSortedY(problem.GetTrainingDataset());

            sort(inputDatasetSortedY.begin(), inputDatasetSortedY.end(),
             [](const Point& point1, const Point& point2)
             {
                 return point1.y < point2.y;
             });

            sort(trainingDatasetSortedY.begin(), trainingDatasetSortedY.end(),
             [](const Point& point1, const Point& point2)
             {
                 return point1.y < point2.y;
             });

            size_t inputDatasetStripeSize = inputDatasetSortedY.size()/numStripes + 1;
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();

            auto inputIterStart = inputDatasetSortedY.cbegin();
            auto inputIterEnd = inputIterStart + inputDatasetStripeSize;
            auto trainingIterStart = trainingDatasetSortedY.cbegin();

            bool exit = false;

            do
            {
                while (inputIterEnd < inputDatasetSortedYEnd && prev(inputIterEnd)->y == inputIterEnd->y)
                {
                    ++inputIterEnd;
                }

                pInputDatasetStripe->push_back(point_vector_t(inputIterStart, inputIterEnd));

                double minY = inputIterStart->y <= trainingIterStart->y ? inputIterStart->y : trainingIterStart->y;

                sort(pInputDatasetStripe->back().begin(), pInputDatasetStripe->back().end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.x < point2.x;
                     });

                double maxY = minY;

                if (trainingIterStart < trainingDatasetSortedYEnd)
                {
                    auto trainingIterEnd = inputIterEnd == inputDatasetSortedYEnd ? trainingDatasetSortedYEnd :
                                            upper_bound(trainingIterStart, trainingDatasetSortedYEnd, prev(inputIterEnd)->y,
                                                          [](const double& value, const Point& point) { return value < point.y; } );

                    pTrainingDatasetStripe->push_back(point_vector_t(trainingIterStart, trainingIterEnd));

                    maxY = prev(trainingIterEnd)->y >= prev(inputIterEnd)->y ? prev(trainingIterEnd)->y : prev(inputIterEnd)->y;

                    sort(pTrainingDatasetStripe->back().begin(), pTrainingDatasetStripe->back().end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.x < point2.x;
                     });

                     trainingIterStart = trainingIterEnd;
                }
                else
                {
                    pTrainingDatasetStripe->push_back(point_vector_t());

                    maxY = prev(inputIterEnd)->y;
                }

                pStripeBoundaries->push_back({minY, maxY});

                if (inputIterEnd < inputDatasetSortedYEnd)
                {
                    inputIterStart = inputIterEnd;
                    if ((size_t)distance(inputIterStart, inputDatasetSortedYEnd) >= inputDatasetStripeSize)
                    {
                        inputIterEnd = inputIterStart + inputDatasetStripeSize;
                    }
                    else
                    {
                        inputIterEnd = inputDatasetSortedYEnd;
                    }
                }
                else
                {
                    exit = true;
                }

            } while (!exit);

            return {*pInputDatasetStripe, *pTrainingDatasetStripe, *pStripeBoundaries};
        }

    protected:

    private:
        unique_ptr<point_vector_vector_t> pInputDatasetStripe;
        unique_ptr<point_vector_vector_t> pTrainingDatasetStripe;
        unique_ptr<vector<StripeBoundaries>> pStripeBoundaries;
};

#endif // ALLKNNRESULTSTRIPES_H
