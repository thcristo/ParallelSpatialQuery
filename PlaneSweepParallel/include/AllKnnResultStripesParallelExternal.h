#ifndef ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
#define ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H

#include "AllKnnResultStripes.h"




class AllKnnResultStripesParallelExternal : public AllKnnResult
{
    public:
        AllKnnResultStripesParallelExternal(const AllKnnProblem& problem, const string& filePrefix) : AllKnnResult(problem, filePrefix)
        {
        }

        AllKnnResultStripesParallelExternal(const AllKnnProblem& problem, const string& filePrefix, bool parallelSort, bool splitByT) : AllKnnResult(problem, filePrefix), splitByT(splitByT), parallelSort(parallelSort)
        {
        }

        virtual ~AllKnnResultStripesParallelExternal() {}

        int SplitStripes()
        {
            //ext_point_vector_t inputDataset
            return 0;
        }
    protected:

    private:
        bool splitByT = false;
        bool parallelSort = false;
};

#endif // ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
