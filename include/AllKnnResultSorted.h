#ifndef ALLKNNRESULTSORTED_H
#define ALLKNNRESULTSORTED_H

#include "AllKnnResult.h"


class AllKnnResultSorted : public AllKnnResult
{
    public:
        AllKnnResultSorted(const AllKnnProblem& problem, const string& filePrefix) : AllKnnResult(problem, filePrefix)
        {
        }

        virtual ~AllKnnResultSorted() {}

        const point_vector_t& GetInputDatasetSorted()
        {
            if (!pInputDatasetSorted)
            {
                pInputDatasetSorted.reset(new point_vector_t(problem.GetInputDataset()));

                sort(pInputDatasetSorted->begin(), pInputDatasetSorted->end(),
                 [](const Point& point1, const Point& point2)
                 {
                     return point1.x < point2.x;
                 });
            }

            return *pInputDatasetSorted;
        }

        const point_vector_t& GetTrainingDatasetSorted()
        {
            if (!pTrainingDatasetSorted)
            {
                pTrainingDatasetSorted.reset(new point_vector_t(problem.GetTrainingDataset()));

                sort(pTrainingDatasetSorted->begin(), pTrainingDatasetSorted->end(),
                 [](const Point& point1, const Point& point2)
                 {
                     return point1.x < point2.x;
                 });
            }

            return *pTrainingDatasetSorted;
        }

        const vector<point_vector_t>& GetTrainingDatasetSortedCopy()
        {
            if (!pTrainingDatasetSortedCopy)
            {
                pTrainingDatasetSortedCopy.reset(new vector<point_vector_t>(8, GetTrainingDatasetSorted()));
            }

            return *pTrainingDatasetSortedCopy;
        }

    protected:

    private:
        unique_ptr<point_vector_t> pInputDatasetSorted;
        unique_ptr<point_vector_t> pTrainingDatasetSorted;
        unique_ptr<point_vector_vector_t> pTrainingDatasetSortedCopy;
};

#endif // ALLKNNRESULTSORTED_H
