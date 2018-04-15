#ifndef STRIPESWINDOW_H
#define STRIPESWINDOW_H


class StripesWindow
{
    public:
        StripesWindow(size_t startStripe, size_t endStripe, unique_ptr<point_vector_vector_t>& pInputStripes,
                      unique_ptr<point_vector_vector_t>& pTrainingStripes, unique_ptr<vector<StripeBoundaries_t>>& pBoundaries,
                      size_t numNeighbors)
                      : startStripe(startStripe), endStripe(endStripe), secondPass(false),
                        pInputDatasetStripe(move(pInputStripes)), pTrainingDatasetStripe(move(pTrainingStripes)),
                        pStripeBoundaries(move(pBoundaries)), numNeighbors(numNeighbors)
        {
            size_t numStripes = GetNumStripes();
            pNeighborsContainer.reset(new pointNeighbors_vector_vector_t(numStripes,
                    vector<PointNeighbors<neighbors_priority_queue_t>>()));

            for(size_t i=0; i < numStripes; ++i)
            {
                size_t numInputPoints = pInputDatasetStripe->at(i).size();
                pNeighborsContainer->at(i).assign(numInputPoints, PointNeighbors<neighbors_priority_queue_t>(numNeighbors));
            }
        }

        StripesWindow(size_t startStripe, size_t endStripe, unique_ptr<point_vector_vector_t>& pTrainingStripes,
                      unique_ptr<vector<StripeBoundaries_t>>& pBoundaries)
                      : startStripe(startStripe), endStripe(endStripe), secondPass(true),
                        pTrainingDatasetStripe(move(pTrainingStripes)),
                        pStripeBoundaries(move(pBoundaries))
        {
        }

        virtual ~StripesWindow() {}

        size_t GetStartStripe() const
        {
           return startStripe;
        }

        size_t GetEndStripe() const
        {
           return endStripe;
        }

        size_t GetNumStripes() const
        {
           return endStripe - startStripe + 1;
        }

        bool IsSecondPass() const
        {
            return secondPass;
        }

        StripeData GetStripeData() const
        {
            return {*pInputDatasetStripe, *pTrainingDatasetStripe, *pStripeBoundaries};
        }

        pointNeighbors_vector_vector_t& GetNeighborsContainer() const
        {
                return *pNeighborsContainer;
        }

    protected:

    private:
        size_t startStripe = 0;
        size_t endStripe = 0;
        bool secondPass = false;
        unique_ptr<point_vector_vector_t> pInputDatasetStripe;
        unique_ptr<point_vector_vector_t> pTrainingDatasetStripe;
        unique_ptr<vector<StripeBoundaries_t>> pStripeBoundaries;
        unique_ptr<pointNeighbors_vector_vector_t> pNeighborsContainer;
        size_t numNeighbors = 0;
};

#endif // STRIPESWINDOW_H
