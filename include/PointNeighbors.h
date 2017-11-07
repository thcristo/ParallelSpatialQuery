#ifndef POINTNEIGHBORS_H
#define POINTNEIGHBORS_H

#include <memory.h>
#include <unordered_map>
#include "PlaneSweepParallel.h"

class PointNeighbors
{
    public:
        virtual bool HasNext() const = 0;
        virtual Neighbor Next() = 0;
        virtual void Add(point_vector_t::const_iterator pointIter, const double distanceSquared) = 0;
        virtual const Neighbor& MaxDistanceElement() const = 0;
};

class PointNeighborsPriorityQueue : public PointNeighbors
{
    public:
        PointNeighborsPriorityQueue(size_t numNeighbors)
        {
            vector<Neighbor> defaultNeighbors(numNeighbors, {nullptr, numeric_limits<double>::max()});
            pContainer.reset(new neighbors_priority_queue_t(defaultNeighbors.cbegin(), defaultNeighbors.cend()));
        }

        virtual ~PointNeighborsPriorityQueue() {}

        bool HasNext() const override
        {
            return !pContainer->empty();
        }

        Neighbor Next() override
        {
            Neighbor neighbor = pContainer->top();
            pContainer->pop();
            return neighbor;
        }

        void Add(point_vector_t::const_iterator pointIter, const double distanceSquared) override
        {
            auto& lastNeighbor = pContainer->top();

            if (distanceSquared < lastNeighbor.distanceSquared)
            {
                pContainer->pop();

                Neighbor newNeighbor = {&*pointIter, distanceSquared};

                pContainer->push(newNeighbor);
            }
        }

        const Neighbor& MaxDistanceElement() const override
        {
            return pContainer->top();
        }

    private:
        unique_ptr<neighbors_priority_queue_t> pContainer;
};

class PointNeighborsVector : public PointNeighbors
{
    public:
        PointNeighborsVector(size_t numNeighbors)
        {
            pContainer.reset(new vector<Neighbor>(numNeighbors, {nullptr, numeric_limits<double>::max()}));
            insertIter = pContainer->end() - 1;
            returnIter = pContainer->cbegin();
        }

        virtual ~PointNeighborsVector() {}

        bool HasNext() const override
        {
            return returnIter < pContainer->cend();
        }

        Neighbor Next() override
        {
            Neighbor neighbor = *returnIter;
            ++returnIter;
            return neighbor;
        }

        inline void Add(point_vector_t::const_iterator pointIter, const double distanceSquared) override
        {
            *insertIter = {&*pointIter, distanceSquared};
            --insertIter;
        }

        const Neighbor& MaxDistanceElement() const override
        {
            return *pContainer->cbegin();
        }

    private:
        unique_ptr<vector<Neighbor>> pContainer;
        vector<Neighbor>::iterator insertIter;
        vector<Neighbor>::const_iterator returnIter;
};

typedef unordered_map<int, unique_ptr<PointNeighbors>> neighbors_container_t;

#endif // POINTNEIGHBORS_H
