#ifndef ABSTRACTALLKNNALGORITHM_H
#define ABSTRACTALLKNNALGORITHM_H

#include <vector>
#include <limits>
#include <memory>
#include <array>
#include "AllKnnProblem.h"
#include "AllKnnResult.h"
//#include "PlaneSweepParallel.h"

#include <tbb/tbb.h>

using namespace tbb;

template<class ProblemT, class ResultT, class PointVectorT, class PointVectorIteratorT, class NeighborsContainerT>
class AbstractAllKnnAlgorithm
{
    public:
        virtual ~AbstractAllKnnAlgorithm() {}
        virtual unique_ptr<ResultT> Process(ProblemT& problem) = 0;
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
        unique_ptr<NeighborsContainerT> CreateNeighborsContainer(const PointVectorT& inputDataset, size_t numNeighbors) const
        {
            try
            {
                unique_ptr<NeighborsContainerT> pContainer(new NeighborsContainerT(inputDataset.size(),
                        PointNeighbors<neighbors_priority_queue_t<PointVectorIteratorT>, PointVectorIteratorT>(numNeighbors),
                        cache_aligned_allocator<PointNeighbors<neighbors_priority_queue_t<PointVectorIteratorT>, PointVectorIteratorT>>()));

                return pContainer;
            }
            catch(bad_alloc)
            {
                throw ApplicationException("Cannot allocate memory for neighbors container.");
            }
        }

        inline void AddNeighbor(PointVectorIteratorT inputPoint, PointVectorIteratorT trainingPoint,
                                 PointNeighbors<neighbors_priority_queue_t<PointVectorIteratorT>, PointVectorIteratorT>& neighbors) const
        {
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint);
            neighbors.Add(trainingPoint, dsq);
        }

        inline bool CheckAddNeighbor(PointVectorIteratorT inputPoint, PointVectorIteratorT trainingPoint,
                                 PointNeighbors<neighbors_priority_queue_t<PointVectorIteratorT>, PointVectorIteratorT>& neighbors) const
        {
            double dx = 0.0;
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint, dx);
            return neighbors.CheckAdd(trainingPoint, dsq, dx);
        }

        inline bool CheckAddNeighbor(PointVectorIteratorT inputPoint, PointVectorIteratorT trainingPoint,
                                 PointNeighbors<neighbors_priority_queue_t<PointVectorIteratorT>, PointVectorIteratorT>& neighbors, const double& mindy) const
        {
            double dx = 0.0;
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint, dx);
            return neighbors.CheckAdd(trainingPoint, dsq, dx, mindy);
        }

        inline void AddNeighbor(PointVectorIteratorT trainingPoint, double distanceSquared,
                                 PointNeighbors<neighbors_priority_queue_t<PointVectorIteratorT>, PointVectorIteratorT>& neighbors) const
        {
            neighbors.Add(trainingPoint, distanceSquared);
        }

        inline double CalcDistanceSquared(PointVectorIteratorT p1, PointVectorIteratorT p2) const
        {
            double dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }

        inline double CalcDistanceSquared(PointVectorIteratorT p1, PointVectorIteratorT p2, double& dx) const
        {
            dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }


        /*
        inline double CalcDistanceSquared(point_vector_iterator_t p1, point_vector_iterator_t p2, double& dx) const
        {
            array<const double*, 2> point1 = { &p1->x, &p1->y };
            array<const double*, 2> point2 = { &p2->x, &p2->y };
            array<double, 2> delta;
            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for (int i=0; i < 2; ++i)
            {
                delta[i] = *point2[i] - *point1[i];
                sum = delta[i]*delta[i];
            }

            dx = delta[0];

            return sum;
        }
        */
        /*
        inline array<double, 2> CalcDistanceSquared(point_vector_iterator_t p1, array<point_vector_iterator_t, 2>& p2, array<double, 2>& dx) const
        {
            array<double, 2> distancesSquared;
            array<double, 2> dy;

            #pragma omp simd
            for (int i=0; i < 2; ++i)
            {
                dx[i] = p2[i]->x - p1->x;
                dy[i] = p2[i]->y - p1->y;
                distancesSquared[i] = dx[i]*dx[i] + dy[i]*dy[i];
            }

            return distancesSquared;
        }
        */

    private:

};

#endif // ABSTRACTALLKNNALGORITHM_H
