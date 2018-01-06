#ifndef POINTNEIGHBORS_H
#define POINTNEIGHBORS_H

#include <memory.h>
#include <unordered_map>
#include "PlaneSweepParallel.h"

#include <tbb/tbb.h>

using namespace tbb;

template<class PointVectorIteratorT>
class NeighborsEnumerator
{
    public:
        virtual bool HasNext() = 0;
        virtual Neighbor<PointVectorIteratorT> Next() = 0;
        virtual void AddAllRemoved(const vector<Neighbor<PointVectorIteratorT>>& neighbors) = 0;
        virtual size_t GetNumAdditions() = 0;
};

template<class Container, class PointVectorIteratorT>
class PointNeighbors : public NeighborsEnumerator<PointVectorIteratorT>
{
    public:
        PointNeighbors(size_t numNeighbors);
        virtual ~PointNeighbors();

        bool HasNext();
        Neighbor<PointVectorIteratorT> Next();

        void AddAllRemoved(const vector<Neighbor<PointVectorIteratorT>>& neighbors);

    private:
        Container container;
        size_t numNeighbors;
        long id;
};

template<class PointVectorIteratorT>
class PointNeighbors<neighbors_priority_queue_t<PointVectorIteratorT>, PointVectorIteratorT> : public NeighborsEnumerator<PointVectorIteratorT>
{
    public:
        PointNeighbors(size_t numNeighbors) : PointNeighbors(neighbors_vector_t<PointVectorIteratorT>(numNeighbors, {PointVectorIteratorT(), numeric_limits<double>::max()}))
        {
        }

        PointNeighbors(neighbors_vector_t<PointVectorIteratorT> neighborsVector) : numNeighbors(neighborsVector.size()),
            container(neighborsVector.cbegin(), neighborsVector.cend())
        {
        }

        /*
        PointNeighbors(size_t numNeighbors) : numNeighbors(numNeighbors)
        {
        }
        */

        virtual ~PointNeighbors() {}

        bool HasNext() override
        {
            return !container.empty();
        }

        Neighbor<PointVectorIteratorT> Next() override
        {
            Neighbor<PointVectorIteratorT> neighbor = container.top();
            container.pop();
            return neighbor;
        }

        inline void Add(PointVectorIteratorT pointIter, const double distanceSquared)
        {
            auto& lastNeighbor = container.top();

            if (distanceSquared < lastNeighbor.distanceSquared)
            {
                container.pop();
                Neighbor<PointVectorIteratorT> newNeighbor = {pointIter, distanceSquared};
                container.push(newNeighbor);
                ++numAdditions;
            }
        }

        void AddAllRemoved(const vector<Neighbor<PointVectorIteratorT>>& neighbors)
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

        inline bool CheckAdd(PointVectorIteratorT pointIter, const double& distanceSquared, const double& dx)
        {
            auto& lastNeighbor = container.top();
            double maxDistance = lastNeighbor.distanceSquared;

            if (distanceSquared < maxDistance)
            {
                container.pop();
                Neighbor<PointVectorIteratorT> newNeighbor = {pointIter, distanceSquared};
                container.push(newNeighbor);
                ++numAdditions;
            }
            else if (dx*dx >= maxDistance)
            {
                return false;
            }

            return true;
        }

        inline bool CheckAdd(PointVectorIteratorT pointIter, const double& distanceSquared, const double& dx, const double& mindy)
        {
            auto& lastNeighbor = container.top();
            double maxDistance = lastNeighbor.distanceSquared;

            if (distanceSquared < maxDistance)
            {
                container.pop();
                Neighbor<PointVectorIteratorT> newNeighbor = {pointIter, distanceSquared};
                container.push(newNeighbor);
                ++numAdditions;
            }
            else if (dx*dx + mindy*mindy >= maxDistance)
            {
                return false;
            }

            return true;
        }

        /*
        inline array<bool,2> CheckAdd(const array<point_vector_iterator_t,2>& pointIter, const array<double,2>& distanceSquared, const array<double,2>& dx)
        {
            array<bool, 2> continuations = {true, true};

            for (int i=0; i < 2; ++i)
            {
                auto& lastNeighbor = container.top();
                double maxDistance = lastNeighbor.distanceSquared;

                if (distanceSquared[i] < maxDistance)
                {
                    container.pop();
                    Neighbor newNeighbor = {&*pointIter[i], distanceSquared[i]};
                    container.push(newNeighbor);
                    ++numAdditions;
                }
                else if (dx[i]*dx[i] >= maxDistance)
                {
                    continuations[i] = false;
                }
            }

            return continuations;
        }
        */

        /*
        inline bool CheckAdd(point_vector_iterator_t pointIter, const double& distanceSquared, const double& dx)
        {

            if (container.size() < numNeighbors)
            {
                Neighbor newNeighbor = {&*pointIter, distanceSquared};
                container.push(newNeighbor);
            }
            else
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
            }

            return true;
        }
        */

        inline void AddNoCheck(PointVectorIteratorT pointIter, const double& distanceSquared)
        {
            container.pop();
            Neighbor<PointVectorIteratorT> newNeighbor = {pointIter, distanceSquared};
            container.push(newNeighbor);
            ++numAdditions;
        }

        inline const Neighbor<PointVectorIteratorT>& MaxDistanceElement() const
        {
            return container.top();
        }

    private:
        size_t numNeighbors = 0;
        neighbors_priority_queue_t<PointVectorIteratorT> container;
        size_t numAdditions = 0;
};


template<class Container>
using pointNeighbors_generic_vector_t = vector<PointNeighbors<Container, point_vector_iterator_t>, cache_aligned_allocator<PointNeighbors<Container, point_vector_iterator_t>>>;

typedef pointNeighbors_generic_vector_t<neighbors_priority_queue_t<point_vector_iterator_t>> pointNeighbors_priority_queue_vector_t;

#endif // POINTNEIGHBORS_H
