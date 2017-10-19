#ifndef ABSTRACTALLKNNALGORITHM_H
#define ABSTRACTALLKNNALGORITHM_H

#include <vector>
#include <limits>
#include <memory>
#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "PlaneSweepParallel.h"

using namespace std;

class AbstractAllKnnAlgorithm
{
    public:
        virtual ~AbstractAllKnnAlgorithm() {}
        virtual unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const = 0;

    protected:
        AbstractAllKnnAlgorithm() {}

        /** \brief Creates the container of nearest neighbors for all points
         *
         * \param numPoints The number of points
         * \param numNeighbors The number of nearest neighbors
         * \return An unordered map of point Ids to multisets of Neighbors
         *
         */

        unique_ptr<neighbors_container_t> CreateNeighborsContainer(const vector<Point>& inputDataset, int numNeighbors) const
        {
            try
            {
                unique_ptr<neighbors_container_t> pContainer(new neighbors_container_t(inputDataset.size()));

                vector<Neighbor> defaultNeighbors(numNeighbors, {nullptr, numeric_limits<double>::max()});

                for (auto point = inputDataset.cbegin(); point != inputDataset.cend(); ++point)
                {
                    pContainer->insert(make_pair(point->id, point_neighbors_t(defaultNeighbors.cbegin(), defaultNeighbors.cend())));
                }

                return pContainer;
            }
            catch(bad_alloc)
            {
                throw ApplicationException("Cannot allocate memory for neighbors container.");
            }

        }
};

#endif // ABSTRACTALLKNNALGORITHM_H
