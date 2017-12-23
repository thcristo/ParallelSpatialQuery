#ifndef ABSTRACTALLKNNALGORITHM_H
#define ABSTRACTALLKNNALGORITHM_H

#include <vector>
#include <limits>
#include <memory>
#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "PlaneSweepParallel.h"

#include <tbb/tbb.h>

using namespace tbb;


template<class OuterContainer>
unique_ptr<OuterContainer> CreateNeighborsContainer(const point_vector_t& inputDataset, size_t numNeighbors)
{

}

/*
template<>
unique_ptr<neighbors_priority_queue_container_t> CreateNeighborsContainer<neighbors_priority_queue_container_t>(const point_vector_t& inputDataset, size_t numNeighbors)
{
    try
    {
        unique_ptr<neighbors_priority_queue_container_t> pContainer(new neighbors_priority_queue_container_t(inputDataset.size()));

        for (auto point = inputDataset.cbegin(); point != inputDataset.cend(); ++point)
        {
            pContainer->insert(make_pair(point->id, PointNeighbors<neighbors_priority_queue_t>(numNeighbors)));
        }

        return pContainer;
    }
    catch(bad_alloc)
    {
        throw ApplicationException("Cannot allocate memory for neighbors container.");
    }
}
*/

template<>
unique_ptr<pointNeighbors_priority_queue_vector_t> CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(const point_vector_t& inputDataset, size_t numNeighbors)
{
    try
    {
        unique_ptr<pointNeighbors_priority_queue_vector_t> pContainer(new pointNeighbors_priority_queue_vector_t(inputDataset.size(),
                                                                    PointNeighbors<neighbors_priority_queue_t>(numNeighbors),
                                                                    cache_aligned_allocator<PointNeighbors<neighbors_priority_queue_t>>()));

        return pContainer;
    }
    catch(bad_alloc)
    {
        throw ApplicationException("Cannot allocate memory for neighbors container.");
    }
}

class AbstractAllKnnAlgorithm
{
    public:
        virtual ~AbstractAllKnnAlgorithm() {}
        virtual unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) = 0;
        virtual string GetTitle() const = 0;
    protected:
        AbstractAllKnnAlgorithm() {}

        /** \brief Creates the container of nearest neighbors for all points
         *
         * \param numPoints The number of points
         * \param numNeighbors The number of nearest neighbors
         * \return An unordered map of point Ids to multisets of Neighbors
         *
         */
        template<class OuterContainer>
        unique_ptr<OuterContainer> CreateNeighborsContainer(const point_vector_t& inputDataset, size_t numNeighbors) const
        {
            return ::CreateNeighborsContainer<OuterContainer>(inputDataset, numNeighbors);
        }

        template<class Container>
        inline void AddNeighbor(point_vector_iterator_t inputPoint, point_vector_iterator_t trainingPoint,
                                 PointNeighbors<Container>& neighbors) const
        {
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint);
            neighbors.Add(trainingPoint, dsq);
        }


        template<class Container>
        inline bool CheckAddNeighbor(point_vector_iterator_t inputPoint, point_vector_iterator_t trainingPoint,
                                 PointNeighbors<Container>& neighbors) const
        {
            double dx = 0.0;
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint, dx);
            return neighbors.CheckAdd(trainingPoint, dsq, dx);
        }


        /*
        template<class Container>
        inline bool CheckAddNeighbor(point_vector_iterator_t inputPoint, point_vector_iterator_t trainingPoint,
                                 PointNeighbors<Container>& neighbors) const
        {
            double dx = trainingPoint->x - inputPoint->x;
            double dxSquared = dx*dx;

            auto& lastNeighbor = neighbors.MaxDistanceElement();
            double maxDistance = lastNeighbor.distanceSquared;

            if (dxSquared >= maxDistance)
            {
                return false;
            }
            else
            {
                double dy = trainingPoint->y - inputPoint->y;
                double dsq = dxSquared + dy*dy;

                if (dsq < maxDistance)
                {
                    neighbors.AddNoCheck(trainingPoint, dsq);
                }
            }

            return true;
        }
        */

        template<class Container>
        inline void AddNeighbor(point_vector_iterator_t trainingPoint, double distanceSquared,
                                 PointNeighbors<Container>& neighbors) const
        {
            neighbors.Add(trainingPoint, distanceSquared);
        }

        inline double CalcDistanceSquared(point_vector_iterator_t p1, point_vector_iterator_t p2) const
        {
            double dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }

        inline double CalcDistanceSquared(point_vector_iterator_t p1, point_vector_iterator_t p2, double& dx) const
        {
            dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }


    private:

};

#endif // ABSTRACTALLKNNALGORITHM_H
