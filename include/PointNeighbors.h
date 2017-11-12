#ifndef POINTNEIGHBORS_H
#define POINTNEIGHBORS_H

#include <memory.h>
#include <unordered_map>
#include "PlaneSweepParallel.h"

class NeighborsEnumerator
{
    public:
        virtual bool HasNext() = 0;
        virtual Neighbor Next() = 0;
};

template<class Container>
class PointNeighbors : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors);
        virtual ~PointNeighbors();

        bool HasNext();
        Neighbor Next();

        inline void Add(point_vector_t::const_iterator pointIter, const double distanceSquared);
        inline const Neighbor& MaxDistanceElement() const;

    private:
        Container container;
        size_t numNeighbors;
};

template<>
class PointNeighbors<neighbors_priority_queue_t> : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors) : PointNeighbors(neighbors_vector_t(numNeighbors, {nullptr, numeric_limits<double>::max()}))
        {
        }

        PointNeighbors(neighbors_vector_t neighborsVector) : numNeighbors(neighborsVector.size()),
            container(neighborsVector.cbegin(), neighborsVector.cend()),
            numAddedNeighbors(0)
        {
        }

        virtual ~PointNeighbors() {}

        bool HasNext() override
        {
            return !container.empty();
        }

        Neighbor Next() override
        {
            Neighbor neighbor = container.top();
            container.pop();
            return neighbor;
        }

        inline void Add(point_vector_t::const_iterator pointIter, const double distanceSquared)
        {
            auto& lastNeighbor = container.top();

            if (distanceSquared < lastNeighbor.distanceSquared)
            {
                container.pop();
                Neighbor newNeighbor = {&*pointIter, distanceSquared};
                container.push(newNeighbor);
            }
        }

        inline const Neighbor& MaxDistanceElement() const
        {
            return container.top();
        }

        inline size_t Size() const
        {
            return numAddedNeighbors;
        }

    private:
        size_t numNeighbors;
        neighbors_priority_queue_t container;
        size_t numAddedNeighbors;

};

template<>
class PointNeighbors<neighbors_deque_t> : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors) : numNeighbors(numNeighbors)
        {
        }

        virtual ~PointNeighbors() {}

        bool HasNext() override
        {
            if (numAddedNeighbors > 0)
            {
                if (!iterationStarted)
                {
                    returnIter = container.cbegin();
                    iterationStarted = true;
                }
                return returnIter < container.cend();
            }
            else
            {
                return false;
            }
        }

        Neighbor Next() override
        {
            Neighbor neighbor = *returnIter;
            ++returnIter;
            return neighbor;
        }

        inline void Add(point_vector_t::const_iterator pointIter, const double distanceSquared)
        {
            container.push_front({&*pointIter, distanceSquared});
            ++numAddedNeighbors;
        }

        inline const Neighbor* MaxDistanceElement() const
        {
            if (numAddedNeighbors > 0)
            {
                return &container[0];
            }
            else
            {
                throw ApplicationException("Cannot get max distance element because no neighbors exist in deque.");
            }

        }

        inline size_t Size() const
        {
            return numAddedNeighbors;
        }

    private:
        size_t numNeighbors;
        neighbors_deque_t container;
        bool iterationStarted;
        neighbors_deque_t::const_iterator returnIter;
        size_t numAddedNeighbors;
};

template<class Container>
using neighbors_container_t = unordered_map<long, PointNeighbors<Container>>;

typedef neighbors_container_t<neighbors_priority_queue_t> neighbors_priority_queue_container_t;
typedef neighbors_container_t<neighbors_deque_t> neighbors_vector_container_t;


#endif // POINTNEIGHBORS_H
