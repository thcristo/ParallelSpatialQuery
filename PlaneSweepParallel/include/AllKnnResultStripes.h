#ifndef ALLKNNRESULTSTRIPES_H
#define ALLKNNRESULTSTRIPES_H

#include "AllKnnResult.h"
#include <tbb/tbb.h>

using namespace tbb;
template<class ProblemT, class NeighborVectorT, class PointVectorT, class PointIdVectorT, class PointVectorIteratorT, class PointVectorVectorT, class StripeBoundariesVectorT>
class AllKnnResultStripes : public AllKnnResult<ProblemT, NeighborVectorT, PointVectorT, PointIdVectorT, PointVectorIteratorT>
{
    using AllKnnResult<ProblemT, NeighborVectorT, PointVectorT, PointIdVectorT, PointVectorIteratorT>::problem;

    public:
        AllKnnResultStripes(const ProblemT& problem, const string& filePrefix)
            : AllKnnResult<ProblemT, NeighborVectorT, PointVectorT, PointIdVectorT, PointVectorIteratorT>(problem, filePrefix)
        {
        }

        AllKnnResultStripes(const ProblemT& problem, const string& filePrefix, bool parallelSort)
            : AllKnnResult<ProblemT, NeighborVectorT, PointVectorT, PointIdVectorT, PointVectorIteratorT>(problem, filePrefix),
                parallelSort(parallelSort)
        {
        }

        virtual ~AllKnnResultStripes() {}

        StripeData<PointVectorVectorT, StripeBoundariesVectorT> GetStripeData(int numStripes)
        {
            if (!pInputDatasetStripe)
            {
                pInputDatasetStripe.reset(new PointVectorVectorT());
            }

            if (!pTrainingDatasetStripe)
            {
                pTrainingDatasetStripe.reset(new PointVectorVectorT());
            }

            if (!pStripeBoundaries)
            {
                pStripeBoundaries.reset(new StripeBoundariesVectorT());
            }

            PointVectorT inputDatasetSortedY(problem.GetInputDataset());
            PointVectorT trainingDatasetSortedY(problem.GetTrainingDataset());

            if (parallelSort)
            {
                parallel_sort(inputDatasetSortedY.begin(), inputDatasetSortedY.end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.y < point2.y;
                     });

                parallel_sort(trainingDatasetSortedY.begin(), trainingDatasetSortedY.end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.y < point2.y;
                     });
            }
            else
            {
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
            }


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

                pInputDatasetStripe->push_back(PointVectorT(inputIterStart, inputIterEnd));

                double minY = inputIterStart->y <= trainingIterStart->y ? inputIterStart->y : trainingIterStart->y;

                if (parallelSort)
                {
                    parallel_sort(pInputDatasetStripe->back().begin(), pInputDatasetStripe->back().end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }
                else
                {
                    sort(pInputDatasetStripe->back().begin(), pInputDatasetStripe->back().end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }

                double maxY = minY;

                if (trainingIterStart < trainingDatasetSortedYEnd)
                {
                    auto trainingIterEnd = inputIterEnd == inputDatasetSortedYEnd ? trainingDatasetSortedYEnd :
                                            upper_bound(trainingIterStart, trainingDatasetSortedYEnd, prev(inputIterEnd)->y,
                                                          [](const double& value, const Point& point) { return value < point.y; } );

                    pTrainingDatasetStripe->push_back(PointVectorT(trainingIterStart, trainingIterEnd));

                    maxY = prev(trainingIterEnd)->y >= prev(inputIterEnd)->y ? prev(trainingIterEnd)->y : prev(inputIterEnd)->y;

                    if (parallelSort)
                    {
                        parallel_sort(pTrainingDatasetStripe->back().begin(), pTrainingDatasetStripe->back().end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                    }
                    else
                    {
                        sort(pTrainingDatasetStripe->back().begin(), pTrainingDatasetStripe->back().end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                    }

                    trainingIterStart = trainingIterEnd;
                }
                else
                {
                    pTrainingDatasetStripe->push_back(PointVectorT());

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
        unique_ptr<PointVectorVectorT> pInputDatasetStripe;
        unique_ptr<PointVectorVectorT> pTrainingDatasetStripe;
        unique_ptr<StripeBoundariesVectorT> pStripeBoundaries;
        bool parallelSort = false;
};

#endif // ALLKNNRESULTSTRIPES_H
