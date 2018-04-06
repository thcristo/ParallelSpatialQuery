#ifndef STRIPESWINDOW_H
#define STRIPESWINDOW_H


class StripesWindow
{
    public:
        StripesWindow() {}
        virtual ~StripesWindow() {}

        int GetStartStripe()
        {
           return -1;
        }

        int GetEndStripe()
        {
           return -1;
        }

        int GetNumStripes()
        {
           return 0;
        }

        bool IsSecondPass()
        {
            return false;
        }

        StripeData GetStripeData()
        {
            return {*pInputDatasetStripe, *pTrainingDatasetStripe, *pStripeBoundaries};
        }

        pointNeighbors_priority_queue_map_t& GetNeighborsContainer()
        {
                return *pNeighborsContainer;
        }

        void CommitWindow()
        {
            //pResult->CheckAddPendingPoint(inputPointIter, neighbors);
        }

    protected:

    private:
        unique_ptr<point_vector_vector_t> pInputDatasetStripe;
        unique_ptr<point_vector_vector_t> pTrainingDatasetStripe;
        unique_ptr<vector<StripeBoundaries_t>> pStripeBoundaries;
        unique_ptr<pointNeighbors_priority_queue_map_t> pNeighborsContainer;
};

#endif // STRIPESWINDOW_H
