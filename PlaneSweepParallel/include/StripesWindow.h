/* Class definition for a window of stripes used by the external memory algorithm */
#ifndef STRIPESWINDOW_H
#define STRIPESWINDOW_H


/** \brief Window of stripes used by external memory algorithm
 */
class StripesWindow
{
    public:

        /** \brief Constructor for a window to be used by first phase of external memory algorithm (input and training points)
         *
         */
        StripesWindow(size_t startStripe, size_t endStripe, std::unique_ptr<point_vector_vector_t>& pInputStripes,
                      std::unique_ptr<point_vector_vector_t>& pTrainingStripes, std::unique_ptr<std::vector<StripeBoundaries_t>>& pBoundaries,
                      size_t numNeighbors)
                      : startStripe(startStripe), endStripe(endStripe), secondPass(false),
                        pInputDatasetStripe(std::move(pInputStripes)), pTrainingDatasetStripe(std::move(pTrainingStripes)),
                        pStripeBoundaries(std::move(pBoundaries)), numNeighbors(numNeighbors)
        {
            size_t numStripes = GetNumStripes();
            pNeighborsContainer.reset(new pointNeighbors_vector_vector_t(numStripes));

            //neighbor vectors are initialized in parallel for all stripes
            #pragma omp parallel for schedule(dynamic)
            for(size_t i=0; i < numStripes; ++i)
            {
                //if we split stripes by using the training dataset, then each stripe may contain different number of input points
                size_t numInputPoints = pInputDatasetStripe->at(i).size();
                if (numInputPoints > 0)
                {
                    auto& neighborsVector = pNeighborsContainer->at(i);
                    neighborsVector.reserve(numInputPoints);

                    //initialize neighbors for all input points of the current stripe with a very large distance
                    for (size_t iPoint=0; iPoint < numInputPoints; ++iPoint)
                        neighborsVector.emplace_back(PointNeighbors<neighbors_priority_queue_t>(numNeighbors));
                }
            }
        }

        /** \brief Constructor for a window to be used by second phase of external memory algorithm (training points only)
         *
         */
        StripesWindow(size_t startStripe, size_t endStripe, std::unique_ptr<point_vector_vector_t>& pTrainingStripes,
                      std::unique_ptr<std::vector<StripeBoundaries_t>>& pBoundaries)
                      : startStripe(startStripe), endStripe(endStripe), secondPass(true),
                        pTrainingDatasetStripe(std::move(pTrainingStripes)),
                        pStripeBoundaries(std::move(pBoundaries))
        {
        }

        virtual ~StripesWindow()
        {
        }

        /** \brief Returns the starting stripe of the window
         *
         * \return size_t
         *
         */
        size_t GetStartStripe() const
        {
           return startStripe;
        }

        /** \brief Returns the ending stripe of the window
         *
         * \return size_t
         *
         */
        size_t GetEndStripe() const
        {
           return endStripe;
        }

        /** \brief Returns the total number of stripes of the window
         *
         * \return size_t
         *
         */
        size_t GetNumStripes() const
        {
           return endStripe - startStripe + 1;
        }

        /** \brief Returns true if this window is to be used by second phase (contains training points only)
         *
         * \return bool
         *
         */
        bool IsSecondPass() const
        {
            return secondPass;
        }

        /** \brief Returns stripe data for the stripes of this window
         *
         * \return StripeData
         *
         */
        StripeData GetStripeData() const
        {
            return {*pInputDatasetStripe, *pTrainingDatasetStripe, *pStripeBoundaries};
        }

        /** \brief Returns the neighbors for all input points of this window
         *
         * \return pointNeighbors_vector_vector_t&
         *
         */
        pointNeighbors_vector_vector_t& GetNeighborsContainer() const
        {
            return *pNeighborsContainer;
        }

    private:
        size_t startStripe = 0;
        size_t endStripe = 0;
        bool secondPass = false;
        std::unique_ptr<point_vector_vector_t> pInputDatasetStripe;
        std::unique_ptr<point_vector_vector_t> pTrainingDatasetStripe;
        std::unique_ptr<std::vector<StripeBoundaries_t>> pStripeBoundaries;
        std::unique_ptr<pointNeighbors_vector_vector_t> pNeighborsContainer;
        size_t numNeighbors = 0;
};

#endif // STRIPESWINDOW_H
