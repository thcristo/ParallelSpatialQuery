#ifndef ALLKNNRESULTSORTED_H
#define ALLKNNRESULTSORTED_H

#include "AllKnnResult.h"
#include <tbb/tbb.h>

using namespace tbb;


class AllKnnResultSorted : public AllKnnResult
{
    public:
        AllKnnResultSorted(const AllKnnProblem& problem, const string& filePrefix) : AllKnnResult(problem, filePrefix)
        {
        }

        AllKnnResultSorted(const AllKnnProblem& problem, const string& filePrefix, bool parallelSort) : AllKnnResult(problem, filePrefix),
            parallelSort(parallelSort)
        {
        }

        virtual ~AllKnnResultSorted() {}

        const point_vector_t& GetInputDatasetSorted()
        {
            if (!pInputDatasetSorted)
            {
                pInputDatasetSorted.reset(new point_vector_t(problem.GetInputDataset()));

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

        const point_vector_t& GetTrainingDatasetSorted()
        {
            if (!pTrainingDatasetSorted)
            {
                pTrainingDatasetSorted.reset(new point_vector_t(problem.GetTrainingDataset()));

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
        unique_ptr<point_vector_t> pInputDatasetSorted;
        unique_ptr<point_vector_t> pTrainingDatasetSorted;
        bool parallelSort = false;
};

#endif // ALLKNNRESULTSORTED_H
