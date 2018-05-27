/* Class definition for the container of neighbors
    Each instance of this class contains the nearest neighbors of an input point
    The neighbors are stored in a max heap (priority queue)
 */
#ifndef POINTNEIGHBORS_H
#define POINTNEIGHBORS_H

#include <memory.h>
#include <unordered_map>
#include "PlaneSweepParallel.h"
#include <tbb/tbb.h>

using namespace tbb;

/** \brief This class is used as an interface for enumerating neighbors
 */
class NeighborsEnumerator
{
    public:
        virtual bool HasNext() = 0;
        virtual Neighbor Next() = 0;
        virtual void AddAllRemoved(const vector<Neighbor>& neighbors) = 0;
        virtual size_t GetNumAdditions() = 0;
};

/** \brief Generic template class used as a neighbors container
 *          The template parameter is the actual structure that holds the neighbors
 */
template<class Container>
class PointNeighbors : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors);
        virtual ~PointNeighbors();

        bool HasNext();
        Neighbor Next();

        void AddAllRemoved(const vector<Neighbor>& neighbors);

    private:
        Container container;
        size_t numNeighbors;
        unsigned long id;
};

/** \brief Template specialization for priority queue as holder of neighbors (priority queue is implemented as max heap)
 */
template<>
class PointNeighbors<neighbors_priority_queue_t> : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors) : PointNeighbors(neighbors_vector_t(numNeighbors, {0, numeric_limits<double>::max()}))
        {
        }

        PointNeighbors(neighbors_vector_t neighborsVector) : numNeighbors(neighborsVector.size()),
            container(neighborsVector.cbegin(), neighborsVector.cend())
        {
        }

        /** \brief Move constructor
         *
         * \param pointNeighbors PointNeighbors&& instance to move
         *
         */
        PointNeighbors(PointNeighbors&& pointNeighbors)
        {
            numNeighbors = pointNeighbors.numNeighbors;
            std::swap(container, pointNeighbors.container);
            numAdditions = pointNeighbors.numAdditions;
            lowStripe = pointNeighbors.lowStripe;
            highStripe = pointNeighbors.highStripe;
        }

        /** \brief Move assignment operator
         *
         * \param pointNeighbors PointNeighbors&&
         * \return PointNeighbors&
         *
         */
        PointNeighbors& operator=(PointNeighbors&& pointNeighbors)
        {
            if (this != &pointNeighbors)
            {
                numNeighbors = pointNeighbors.numNeighbors;
                std::swap(container, pointNeighbors.container);
                numAdditions = pointNeighbors.numAdditions;
                lowStripe = pointNeighbors.lowStripe;
                highStripe = pointNeighbors.highStripe;
            }

            return *this;
        }

        virtual ~PointNeighbors() {}

        /** \brief Returns true if the heap contains more neighbors
         *
         * \return bool
         *
         */
        bool HasNext() override
        {
            return !container.empty();
        }

        /** \brief Pops the next neighbor from the heap
         *
         * \return Neighbor
         *
         */
        Neighbor Next() override
        {
            Neighbor neighbor = container.top();
            container.pop();
            return neighbor;
        }

        /** \brief Checks neighbor distance with top of the heap and adds the neighbor to the heap
         *
         * \param pointIter point_vector_iterator_t point to add
         * \param distanceSquared const double squared distance calculated by the caller
         * \return void
         *
         */
        inline void Add(point_vector_iterator_t pointIter, const double distanceSquared)
        {
            auto& lastNeighbor = container.top();

            if (distanceSquared < lastNeighbor.distanceSquared)
            {
                container.pop();
                Neighbor newNeighbor = {pointIter->id, distanceSquared};
                container.push(newNeighbor);
                //number of heap additions is recorded for reporting purposes
                ++numAdditions;
            }
        }

        /** \brief Re-inserts a list of neighbors to the heap
         *
         * \param neighbors const vector<Neighbor>&
         * \return void
         *
         */
        void AddAllRemoved(const vector<Neighbor>& neighbors)
        {
            for (int i = neighbors.size() - 1; i >= 0; --i)
            {
                container.push(neighbors[i]);
            }
        }

        size_t GetNumAdditions()
        {
            return numAdditions;
        }

        /** \brief Checks neighbor distance with top of the heap and adds the neighbor to the heap
         *
         * \param pointIter point_vector_iterator_t point to add
         * \param distanceSquared const double& squared distance calculated by the caller
         * \param dx const double& distance difference in x axis calculated by the caller
         * \return bool false if the caller should stop further examination of training points in the same direction
         *
         */
        inline bool CheckAdd(point_vector_iterator_t pointIter, const double& distanceSquared, const double& dx)
        {
            auto& lastNeighbor = container.top();
            double maxDistance = lastNeighbor.distanceSquared;

            if (distanceSquared < maxDistance)
            {
                container.pop();
                Neighbor newNeighbor = {pointIter->id, distanceSquared};
                container.push(newNeighbor);
                ++numAdditions;
            }
            else if (dx*dx >= maxDistance)
            {
                return false;
            }

            return true;
        }

        /** \brief Checks neighbor distance with top of the heap and adds the neighbor to the heap
         *          It takes into account an additional distance for comparisons
         * \param pointIter point_vector_iterator_t point to add
         * \param distanceSquared const double& squared distance calculated by the caller
         * \param dx const double& distance difference in x axis calculated by the caller
         * \param mindy const double& distance in y axis between the input point and the nearest boundary of the stripe
         * \return bool false if the caller should stop further examination of training points in the same direction
         *
         */
        inline bool CheckAdd(point_vector_iterator_t pointIter, const double& distanceSquared, const double& dx, const double& mindy)
        {
            auto& lastNeighbor = container.top();
            double maxDistance = lastNeighbor.distanceSquared;

            if (distanceSquared < maxDistance)
            {
                container.pop();
                Neighbor newNeighbor = {pointIter->id, distanceSquared};
                container.push(newNeighbor);
                ++numAdditions;
            }
            else if (dx*dx + mindy*mindy >= maxDistance)
            {
                return false;
            }

            return true;
        }

        /** \brief Adds the neighbor to the heap without checking distance
         *
         * \param pointIter point_vector_iterator_t point to add
         * \param distanceSquared const double squared distance calculated by the caller
         * \return void
         *
         */
        inline void AddNoCheck(point_vector_iterator_t pointIter, const double& distanceSquared)
        {
            container.pop();
            Neighbor newNeighbor = {pointIter->id, distanceSquared};
            container.push(newNeighbor);
            ++numAdditions;
        }

        /** \brief Returns top of the max heap
         *
         * \return const Neighbor&
         *
         */
        inline const Neighbor& MaxDistanceElement() const
        {
            return container.top();
        }

        /** \brief Sets the lowest stripe searched so far (used by the external memory algorithm)
         *
         * \param stripe size_t
         * \return void
         *
         */
        void setLowStripe(size_t stripe)
        {
            lowStripe = stripe;
        }

        /** \brief Sets the highest stripe searched so far (used by the external memory algorithm)
         *
         * \param stripe size_t
         * \return void
         *
         */
        void setHighStripe(size_t stripe)
        {
            highStripe = stripe;
        }

        size_t getLowStripe() const
        {
            return lowStripe;
        }

        size_t getHighStripe() const
        {
            return highStripe;
        }

    private:
        size_t numNeighbors = 0;
        neighbors_priority_queue_t container;
        size_t numAdditions = 0;
        size_t lowStripe = numeric_limits<size_t>::max();
        size_t highStripe = 0;
};

//type definitions for neighbor containers
template<class Container>
using pointNeighbors_generic_map_t = unordered_map<unsigned long, PointNeighbors<Container>>;

template<class Container>
using pointNeighbors_generic_vector_t = vector<PointNeighbors<Container>, cache_aligned_allocator<PointNeighbors<Container>>>;

typedef pointNeighbors_generic_map_t<neighbors_priority_queue_t> pointNeighbors_priority_queue_map_t;
typedef pointNeighbors_generic_vector_t<neighbors_priority_queue_t> pointNeighbors_priority_queue_vector_t;
typedef vector<vector<PointNeighbors<neighbors_priority_queue_t>>> pointNeighbors_vector_vector_t;
#endif // POINTNEIGHBORS_H
