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
            container(neighborsVector.cbegin(), neighborsVector.cend())
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

        inline bool CheckAdd(point_vector_t::const_iterator pointIter, const double distanceSquared, const double dx)
        {
            auto& lastNeighbor = container.top();
            double maxDistance = lastNeighbor.distanceSquared;

            if (distanceSquared < maxDistance)
            {
                container.pop();
                Neighbor newNeighbor = {&*pointIter, distanceSquared};
                container.push(newNeighbor);
            }
            else if (dx*dx >= maxDistance)
            {
                return false;
            }

            return true;
        }

        inline const Neighbor& MaxDistanceElement() const
        {
            return container.top();
        }

    private:
        size_t numNeighbors;
        neighbors_priority_queue_t container;
};

template<>
class PointNeighbors<neighbors_deque_t> : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors) : numNeighbors(numNeighbors), container(numNeighbors, {nullptr, numeric_limits<double>::max()})
        {
            returnPos = 0;
            insertPos = numNeighbors - 1;
        }

        virtual ~PointNeighbors() {}

        bool HasNext() override
        {
            return returnPos < numNeighbors;
        }

        Neighbor Next() override
        {
            Neighbor neighbor = container[returnPos];
            ++returnPos;
            return neighbor;
        }

        inline void Add(point_vector_t::const_iterator pointIter, const double distanceSquared)
        {
            container[insertPos] = {&*pointIter, distanceSquared};
            --insertPos;
        }

    private:
        size_t numNeighbors;
        neighbors_deque_t container;
        size_t returnPos;
        size_t insertPos;
};

template<class Container>
using neighbors_container_t = unordered_map<long, PointNeighbors<Container>>;

typedef neighbors_container_t<neighbors_priority_queue_t> neighbors_priority_queue_container_t;
typedef neighbors_container_t<neighbors_deque_t> neighbors_deque_container_t;


#endif // POINTNEIGHBORS_H
