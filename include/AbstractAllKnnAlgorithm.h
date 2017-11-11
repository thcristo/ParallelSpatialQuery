#ifndef ABSTRACTALLKNNALGORITHM_H
#define ABSTRACTALLKNNALGORITHM_H

#include <vector>
#include <limits>
#include <memory>
#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "PlaneSweepParallel.h"

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
        template<class Container>
        unique_ptr<neighbors_container_t<Container>> CreateNeighborsContainer(const point_vector_t& inputDataset, size_t numNeighbors) const
        {
            try
            {
                unique_ptr<neighbors_container_t<Container>> pContainer(new neighbors_container_t<Container>(inputDataset.size()));

                for (auto point = inputDataset.cbegin(); point != inputDataset.cend(); ++point)
                {
                    pContainer->insert(make_pair(point->id, PointNeighbors<Container>(numNeighbors)));
                }

                return pContainer;
            }
            catch(bad_alloc)
            {
                throw ApplicationException("Cannot allocate memory for neighbors container.");
            }

        }

        template<class Container>
        inline void AddNeighbor(point_vector_t::const_iterator inputPoint, point_vector_t::const_iterator trainingPoint,
                                 PointNeighbors<Container>& neighbors) const
        {
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint);
            neighbors.Add(trainingPoint, dsq);
        }

        template<class Container>
        inline void AddNeighbor(point_vector_t::const_iterator trainingPoint, double distanceSquared,
                                 PointNeighbors<Container>& neighbors) const
        {
            neighbors.Add(trainingPoint, distanceSquared);
        }

        inline double CalcDistanceSquared(point_vector_t::const_iterator p1, point_vector_t::const_iterator p2) const
        {
            double dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }
};

#endif // ABSTRACTALLKNNALGORITHM_H
