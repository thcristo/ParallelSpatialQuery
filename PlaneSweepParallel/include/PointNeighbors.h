#ifndef POINTNEIGHBORS_H
#define POINTNEIGHBORS_H

#include <memory.h>
#include <unordered_map>
#include "PlaneSweepParallel.h"
#include <tbb/tbb.h>

using namespace tbb;

class NeighborsEnumerator
{
    public:
        virtual bool HasNext() = 0;
        virtual Neighbor Next() = 0;
        virtual void AddAllRemoved(const vector<Neighbor>& neighbors) = 0;
        virtual size_t GetNumAdditions() = 0;
};

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
        long id;
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

        inline void Add(point_vector_iterator_t pointIter, const double distanceSquared)
        {
            auto& lastNeighbor = container.top();

            if (distanceSquared < lastNeighbor.distanceSquared)
            {
                container.pop();
                Neighbor newNeighbor = {&*pointIter, distanceSquared};
                container.push(newNeighbor);
                ++numAdditions;
            }
        }

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

        inline bool CheckAdd(point_vector_iterator_t pointIter, const double& distanceSquared, const double& dx)
        {
            auto& lastNeighbor = container.top();
            double maxDistance = lastNeighbor.distanceSquared;

            if (distanceSquared < maxDistance)
            {
                container.pop();
                Neighbor newNeighbor = {&*pointIter, distanceSquared};
                container.push(newNeighbor);
                ++numAdditions;
            }
            else if (dx*dx >= maxDistance)
            {
                return false;
            }

            return true;
        }

        inline bool CheckAdd(point_vector_iterator_t pointIter, const double& distanceSquared, const double& dx, const double& mindy)
        {
            auto& lastNeighbor = container.top();
            double maxDistance = lastNeighbor.distanceSquared;

            if (distanceSquared < maxDistance)
            {
                container.pop();
                Neighbor newNeighbor = {&*pointIter, distanceSquared};
                container.push(newNeighbor);
                ++numAdditions;
            }
            else if (dx*dx + mindy*mindy >= maxDistance)
            {
                return false;
            }

            return true;
        }

        inline void AddNoCheck(point_vector_iterator_t pointIter, const double& distanceSquared)
        {
            container.pop();
            Neighbor newNeighbor = {&*pointIter, distanceSquared};
            container.push(newNeighbor);
            ++numAdditions;
        }

        inline const Neighbor& MaxDistanceElement() const
        {
            return container.top();
        }

        void setLowStripe(size_t stripe)
        {
            lowStripe = stripe;
        }

        void setHighStripe(size_t stripe)
        {
            highStripe = stripe;
        }

    private:
        size_t numNeighbors = 0;
        neighbors_priority_queue_t container;
        size_t numAdditions = 0;
        size_t lowStripe = -1;
        size_t highStripe = -1;
};

template<class Container>
using pointNeighbors_generic_map_t = unordered_map<long, PointNeighbors<Container>>;

template<class Container>
using pointNeighbors_generic_vector_t = vector<PointNeighbors<Container>, cache_aligned_allocator<PointNeighbors<Container>>>;

typedef pointNeighbors_generic_map_t<neighbors_priority_queue_t> pointNeighbors_priority_queue_map_t;
typedef pointNeighbors_generic_vector_t<neighbors_priority_queue_t> pointNeighbors_priority_queue_vector_t;
typedef vector<vector<PointNeighbors<neighbors_priority_queue_t>>> pointNeighbors_vector_vector_t;
#endif // POINTNEIGHBORS_H
