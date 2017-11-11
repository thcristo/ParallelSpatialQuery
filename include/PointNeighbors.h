#ifndef POINTNEIGHBORS_H
#define POINTNEIGHBORS_H

#include <memory.h>
#include <unordered_map>
#include "PlaneSweepParallel.h"

class NeighborsEnumerator
{
    public:
        virtual bool HasNext() const = 0;
        virtual Neighbor Next() = 0;
};

template<class Container>
class PointNeighbors : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors);
        virtual ~PointNeighbors();

        bool HasNext() const;
        Neighbor Next();

        void Add(point_vector_t::const_iterator pointIter, const double distanceSquared);
        const Neighbor& MaxDistanceElement() const;

    private:
        Container container;
};

template<>
class PointNeighbors<neighbors_priority_queue_t> : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors) : PointNeighbors(neighbors_vector_t(numNeighbors, {nullptr, numeric_limits<double>::max()}))
        {
        }

        PointNeighbors(neighbors_vector_t defaultNeighbors) : container(defaultNeighbors.cbegin(), defaultNeighbors.cend())
        {
        }

        virtual ~PointNeighbors() {}

        bool HasNext() const override
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

    private:
        neighbors_priority_queue_t container;
};

template<>
class PointNeighbors<neighbors_vector_t> : public NeighborsEnumerator
{
    public:
        PointNeighbors(size_t numNeighbors) : container(numNeighbors, {nullptr, numeric_limits<double>::max()})
        {
            insertIter = container.end() - 1;
            returnIter = container.cbegin();
        }

        virtual ~PointNeighbors() {}

        bool HasNext() const override
        {
            return returnIter < container.cend();
        }

        Neighbor Next() override
        {
            Neighbor neighbor = *returnIter;
            ++returnIter;
            return neighbor;
        }

        inline void Add(point_vector_t::const_iterator pointIter, const double distanceSquared)
        {
            *insertIter = {&*pointIter, distanceSquared};
            --insertIter;
        }

        inline const Neighbor& MaxDistanceElement() const
        {
            return *container.cbegin();
        }

    private:
        neighbors_vector_t container;
        neighbors_vector_t::iterator insertIter;
        neighbors_vector_t::const_iterator returnIter;
};

template<class Container>
using neighbors_container_t = unordered_map<long, PointNeighbors<Container>>;

typedef neighbors_container_t<neighbors_priority_queue_t> neighbors_priority_queue_container_t;
typedef neighbors_container_t<neighbors_vector_t> neighbors_vector_container_t;


#endif // POINTNEIGHBORS_H
