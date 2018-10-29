/* This file contains the class definition of AkNN result when the algorithm requires sorting of input and training datasets */

#ifndef ALLKNNRESULTSORTED_H
#define ALLKNNRESULTSORTED_H

#include "AllKnnResult.h"
#include <tbb/tbb.h>

/** \brief Class definition of AkNN result for algorithms that require sorting
 */
class AllKnnResultSorted : public AllKnnResult
{
    public:
        AllKnnResultSorted(const AllKnnProblem& problem, const std::string& filePrefix) : AllKnnResult(problem, filePrefix)
        {
        }

        AllKnnResultSorted(const AllKnnProblem& problem, const std::string& filePrefix, bool parallelSort) : AllKnnResult(problem, filePrefix),
            parallelSort(parallelSort)
        {
        }

        virtual ~AllKnnResultSorted() {}

        /** \brief Returns a copy of input dataset sorted
         *
         * \return point_vector_t& sorted input dataset
         *
         */
        const point_vector_t& GetInputDatasetSorted()
        {
            if (!pInputDatasetSorted)
            {
                //makes a copy of the original dataset
                pInputDatasetSorted.reset(new point_vector_t(problem.GetInputDataset()));

                if (parallelSort)
                {
                    //parallel sort by using the Intel TBB sort routine
                    tbb::parallel_sort(pInputDatasetSorted->begin(), pInputDatasetSorted->end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }
                else
                {
                    //serial sort by using the STL sort routine
                    sort(pInputDatasetSorted->begin(), pInputDatasetSorted->end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }
            }

            return *pInputDatasetSorted;
        }

        /** \brief Returns a copy of training dataset sorted
         *
         * \return point_vector_t& sorted training dataset
         *
         */
        const point_vector_t& GetTrainingDatasetSorted()
        {
            if (!pTrainingDatasetSorted)
            {
                pTrainingDatasetSorted.reset(new point_vector_t(problem.GetTrainingDataset()));

                if (parallelSort)
                {
                    tbb::parallel_sort(pTrainingDatasetSorted->begin(), pTrainingDatasetSorted->end(),
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
        std::unique_ptr<point_vector_t> pInputDatasetSorted;
        std::unique_ptr<point_vector_t> pTrainingDatasetSorted;
        bool parallelSort = false;
};

#endif // ALLKNNRESULTSORTED_H
