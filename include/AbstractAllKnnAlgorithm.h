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
        virtual unique_ptr<PointNeighbors> CreatePointNeighbors(size_t numNeighbors) const = 0;

        /** \brief Creates the container of nearest neighbors for all points
         *
         * \param numPoints The number of points
         * \param numNeighbors The number of nearest neighbors
         * \return An unordered map of point Ids to multisets of Neighbors
         *
         */

        unique_ptr<neighbors_container_t> CreateNeighborsContainer(const point_vector_t& inputDataset, size_t numNeighbors) const
        {
            try
            {
                unique_ptr<neighbors_container_t> pContainer(new neighbors_container_t(inputDataset.size()));

                for (auto point = inputDataset.cbegin(); point != inputDataset.cend(); ++point)
                {
                    pContainer->insert(make_pair(point->id, CreatePointNeighbors(numNeighbors)));
                }

                return pContainer;
            }
            catch(bad_alloc)
            {
                throw ApplicationException("Cannot allocate memory for neighbors container.");
            }

        }

        void AddNeighbor(point_vector_t::const_iterator inputPoint, point_vector_t::const_iterator trainingPoint,
                                 unique_ptr<PointNeighbors>& pNeighbors) const
        {
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint);
            pNeighbors->Add(trainingPoint, dsq);
        }

        void AddNeighbor(point_vector_t::const_iterator trainingPoint, double distanceSquared,
                                 unique_ptr<PointNeighbors>& pNeighbors) const
        {
            pNeighbors->Add(trainingPoint, distanceSquared);
        }

        inline double CalcDistanceSquared(point_vector_t::const_iterator p1, point_vector_t::const_iterator p2) const
        {
            double dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }
};

#endif // ABSTRACTALLKNNALGORITHM_H
