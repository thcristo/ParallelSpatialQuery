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

        unique_ptr<neighbors_container_t> CreateNeighborsContainer(const point_vector_t& inputDataset, int numNeighbors) const
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

        void CheckInsertNeighbor(point_vector_t::const_iterator inputPoint, point_vector_t::const_iterator trainingPoint,
                                 point_neighbors_t& neighbors) const
        {
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint);

            const auto& lastNeighbor = neighbors.top();

            if (dsq < lastNeighbor.distanceSquared)
            {
                neighbors.pop();

                Neighbor newNeighbor = {&*trainingPoint, dsq};

                neighbors.push(newNeighbor);
            }
        }

        inline double CalcDistanceSquared(point_vector_t::const_iterator p1, point_vector_t::const_iterator p2) const
        {
            double dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }
};

#endif // ABSTRACTALLKNNALGORITHM_H
