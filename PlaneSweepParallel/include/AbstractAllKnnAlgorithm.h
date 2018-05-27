/*
    This file contains the definition of an abstract AkNN algorithm class
    Concrete algorithms will derive from this class
 */
#ifndef ABSTRACTALLKNNALGORITHM_H
#define ABSTRACTALLKNNALGORITHM_H

#include <vector>
#include <limits>
#include <memory>
#include <array>
#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "PlaneSweepParallel.h"
#include <tbb/tbb.h>

using namespace tbb;


template<class OuterContainer>
unique_ptr<OuterContainer> CreateNeighborsContainer(const point_vector_t& inputDataset, size_t numNeighbors)
{

}

template<>
unique_ptr<pointNeighbors_priority_queue_vector_t> CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(const point_vector_t& inputDataset, size_t numNeighbors)
{
    try
    {
        //we allocate memory for the list of neighbors of each input point
        //all memory is allocated at once to avoid allocation overhead
        //this function is called only for internal memory algorithms
        unique_ptr<pointNeighbors_priority_queue_vector_t> pContainer(new pointNeighbors_priority_queue_vector_t(cache_aligned_allocator<PointNeighbors<neighbors_priority_queue_t>>()));
        pContainer->reserve(inputDataset.size());

        //for each input point, create a max heap filled with k neighbors of a very large distance
        for (size_t i=0; i < inputDataset.size(); ++i)
            pContainer->emplace_back(PointNeighbors<neighbors_priority_queue_t>(numNeighbors));

        return pContainer;
    }
    catch(bad_alloc)
    {
        throw ApplicationException("Cannot allocate memory for neighbors container.");
    }
}

/**
 * Abstract class for AkNN algorithm
 */
class AbstractAllKnnAlgorithm
{
    public:
        virtual ~AbstractAllKnnAlgorithm() {}

        /** \brief The processing method of the algorithm
         *
         * \param problem AllKnnProblem& The definition of AkNN problem
         * \return unique_ptr<AllKnnResult> A smart pointer to the result of the algorithm
         *
         */
        virtual unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) = 0;

        /** \brief Returns the title of the algorithm
         *
         * \return string
         *
         */
        virtual string GetTitle() const = 0;

        /** \brief
         *
         * \return bool True if this algorithm uses external memory
         *
         */
        virtual bool UsesExternalMemory()
        {
            return false;
        }
    protected:
        AbstractAllKnnAlgorithm() {}

        /** \brief Allocates the container of neighbors for all input points
         *
         * \param inputDataset const point_vector_t& The input dataset
         * \param numNeighbors size_t The number of neighbors
         * \return unique_ptr<OuterContainer> The container of nearest neighbors for each input point
         */
        template<class OuterContainer>
        unique_ptr<OuterContainer> CreateNeighborsContainer(const point_vector_t& inputDataset, size_t numNeighbors) const
        {
            return ::CreateNeighborsContainer<OuterContainer>(inputDataset, numNeighbors);
        }

         /** \brief Adds a training point to the max heap of neighbors for a specific input point
         *
         * \param inputPoint point_vector_iterator_t The input point
         * \param trainingPoint point_vector_iterator_t The training point
         * \param neighbors PointNeighbors<Container>& The max heap of neighbors for the specific input point
         */
        template<class Container>
        inline void AddNeighbor(point_vector_iterator_t inputPoint, point_vector_iterator_t trainingPoint,
                                 PointNeighbors<Container>& neighbors) const
        {
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint);
            neighbors.Add(trainingPoint, dsq);
        }

        /** \brief Adds a training point to the max heap of neighbors for a specific input point
         *
         * \param inputPoint point_vector_iterator_t The input point
         * \param trainingPoint point_vector_iterator_t The training point
         * \param neighbors PointNeighbors<Container>& The max heap of neighbors for the specific input point
         * \return bool False if the squared dx is greater or equal to the top element of the max heap
         */
        template<class Container>
        inline bool CheckAddNeighbor(point_vector_iterator_t inputPoint, point_vector_iterator_t trainingPoint,
                                 PointNeighbors<Container>& neighbors) const
        {
            double dx = 0.0;
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint, dx);
            return neighbors.CheckAdd(trainingPoint, dsq, dx);
        }

        /** \brief Adds a training point to the max heap of neighbors for a specific input point
         *
         * \param inputPoint point_vector_iterator_t The input point
         * \param trainingPoint point_vector_iterator_t The training point
         * \param neighbors PointNeighbors<Container>& The max heap of neighbors for the specific input point
         * \param mindy const double& the dy distance of the input point from the boundary of the stripe where training point belongs to
         * \return bool False if the squared dx is greater or equal to the top element of the max heap plus the squared mindy
         */
        template<class Container>
        inline bool CheckAddNeighbor(point_vector_iterator_t inputPoint, point_vector_iterator_t trainingPoint,
                                 PointNeighbors<Container>& neighbors, const double& mindy) const
        {
            double dx = 0.0;
            double dsq = CalcDistanceSquared(inputPoint, trainingPoint, dx);
            return neighbors.CheckAdd(trainingPoint, dsq, dx, mindy);
        }

        /** \brief Adds a training point to the max heap of neighbors for a specific input point with a pre-calculated distance
         *
         * \param trainingPoint point_vector_iterator_t The training point
         * \param distanceSquared double The squared distance
         * \param neighbors PointNeighbors<Container>& The max heap of neighbors for the specific input point
         */
        template<class Container>
        inline void AddNeighbor(point_vector_iterator_t trainingPoint, double distanceSquared,
                                 PointNeighbors<Container>& neighbors) const
        {
            neighbors.Add(trainingPoint, distanceSquared);
        }

        /** \brief Calculates the squared euclidean distance between two points
         *
         * \param p1 point_vector_iterator_t point 1
         * \param p2 point_vector_iterator_t point 2
         * \return double The squared euclidean distance
         */
        inline double CalcDistanceSquared(point_vector_iterator_t p1, point_vector_iterator_t p2) const
        {
            double dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }

        /** \brief Calculates the squared euclidean distance between two points
         *
         * \param p1 point_vector_iterator_t point 1
         * \param p2 point_vector_iterator_t point 2
         * \param dx double& A reference to the dx to be returned to the caller function
         * \return double The squared euclidean distance
         */
        inline double CalcDistanceSquared(point_vector_iterator_t p1, point_vector_iterator_t p2, double& dx) const
        {
            dx = p2->x - p1->x;
            double dy = p2->y - p1->y;

            return dx*dx + dy*dy;
        }
};

#endif // ABSTRACTALLKNNALGORITHM_H
