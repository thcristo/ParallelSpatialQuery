#ifndef ALLKNNRESULTSORTED_H
#define ALLKNNRESULTSORTED_H

#include "AllKnnResult.h"
#include <tbb/tbb.h>

using namespace tbb;

template<class ProblemT, class NeighborVectorT, class PointVectorT, class PointIdVectorT, class PointVectorIteratorT>
class AllKnnResultSorted : public AllKnnResult<ProblemT, NeighborVectorT, PointVectorT, PointIdVectorT, PointVectorIteratorT>
{
    using AllKnnResult<ProblemT, NeighborVectorT, PointVectorT, PointIdVectorT, PointVectorIteratorT>::problem;

    public:
        AllKnnResultSorted(const ProblemT& problem, const string& filePrefix)
            : AllKnnResult<ProblemT, NeighborVectorT, PointVectorT, PointIdVectorT, PointVectorIteratorT>(problem, filePrefix)
        {
        }

        AllKnnResultSorted(const ProblemT& problem, const string& filePrefix, bool parallelSort)
            : AllKnnResult<ProblemT, NeighborVectorT, PointVectorT, PointIdVectorT, PointVectorIteratorT>(problem, filePrefix),
                    parallelSort(parallelSort)
        {
        }

        virtual ~AllKnnResultSorted() {}

        const PointVectorT& GetInputDatasetSorted()
        {
            if (!pInputDatasetSorted)
            {
                pInputDatasetSorted.reset(new PointVectorT(problem.GetInputDataset()));

                if (parallelSort)
                {
                    parallel_sort(pInputDatasetSorted->begin(), pInputDatasetSorted->end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }
                else
                {
                    sort(pInputDatasetSorted->begin(), pInputDatasetSorted->end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }
            }

            return *pInputDatasetSorted;
        }

        const PointVectorT& GetTrainingDatasetSorted()
        {
            if (!pTrainingDatasetSorted)
            {
                pTrainingDatasetSorted.reset(new PointVectorT(problem.GetTrainingDataset()));

                if (parallelSort)
                {
                    parallel_sort(pTrainingDatasetSorted->begin(), pTrainingDatasetSorted->end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }
                else
                {
                    sort(pTrainingDatasetSorted->begin(), pTrainingDatasetSorted->end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }
            }

            return *pTrainingDatasetSorted;
        }

    protected:

    private:
        unique_ptr<PointVectorT> pInputDatasetSorted;
        unique_ptr<PointVectorT> pTrainingDatasetSorted;
        bool parallelSort = false;
};

#endif // ALLKNNRESULTSORTED_H
