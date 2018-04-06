#ifndef ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
#define ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H

#include "AllKnnResultStripes.h"
#include "StripesWindow.h"



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

        int SplitStripes(int numStripes)
        {
            //ext_point_vector_t inputDataset
            return 0;
        }

        unique_ptr<StripesWindow> GetWindow(int fromStripe, bool secondPass)
        {
            unique_ptr<StripesWindow> pWindow(new StripesWindow());
            return pWindow;
        }

        bool HasAllocationError()
        {
            return false;
        }

        unique_ptr<point_vector_t> GetPendingPointsForWindow(const StripesWindow& window)
        {
            unique_ptr<point_vector_t> pPendingPoints(new point_vector_t());
            return pPendingPoints;
        }

        pointNeighbors_priority_queue_map_t& GetPendingNeighborsContainer()
        {
            return *pPendingNeighborsContainer;
        }

    protected:

    private:
        bool splitByT = false;
        bool parallelSort = false;
        unique_ptr<pointNeighbors_priority_queue_map_t> pPendingNeighborsContainer;
};

#endif // ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
