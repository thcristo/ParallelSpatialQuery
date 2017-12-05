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

            size_t inputDatasetStripeSize = inputDatasetSortedY.size()/numStripes;
            auto trainingIterStart = trainingDatasetSortedY.cbegin();

            for (int i=0; i < numStripes; ++i)
            {
                auto inputIterStart = inputDatasetSortedY.cbegin() + i*inputDatasetStripeSize;
                auto inputIterEnd = (i == numStripes - 1 ? inputDatasetSortedY.cend() : inputIterStart + inputDatasetStripeSize);

                auto trainingIterEnd = (i == numStripes - 1 ? trainingDatasetSortedY.cend() :
                     upper_bound(trainingIterStart, trainingDatasetSortedY.cend(), prev(inputIterEnd)->y,
                                                          [](const double& value, const Point& point) { return value < point.y; } ));

                pInputDatasetStripe->push_back(point_vector_t(inputIterStart, inputIterEnd));

                sort(pInputDatasetStripe->back().begin(), pInputDatasetStripe->back().end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.x < point2.x;
                     });

                if (trainingIterStart != trainingDatasetSortedY.cend())
                {
                    pTrainingDatasetStripe->push_back(point_vector_t(trainingIterStart, trainingIterEnd));

                    sort(pTrainingDatasetStripe->back().begin(), pTrainingDatasetStripe->back().end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.x < point2.x;
                     });
                }
                else
                {
                    pTrainingDatasetStripe->push_back(point_vector_t());
                }

                pStripeBoundaries->push_back({inputIterStart->y, prev(inputIterEnd)->y});
                trainingIterStart = trainingIterEnd;
            }

            return {*pInputDatasetStripe, *pTrainingDatasetStripe, *pStripeBoundaries};
        }

    protected:

    private:
        unique_ptr<point_vector_vector_t> pInputDatasetStripe;
        unique_ptr<point_vector_vector_t> pTrainingDatasetStripe;
        unique_ptr<vector<StripeBoundaries>> pStripeBoundaries;
};

#endif // ALLKNNRESULTSTRIPES_H
