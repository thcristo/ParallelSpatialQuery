#ifndef ABSTRACTALLKNNALGORITHM_H
#define ABSTRACTALLKNNALGORITHM_H

#include <set>
#include <unordered_map>
#include "AllKnnProblem.h"
#include "AllKnnResult.h"

using namespace std;

class PointComparer
{
    public :
        bool operator()(const Neighbor& n1, const Neighbor& n2) const
        {
            return n1.distance < n2.distance;
        }
};

typedef multiset<Neighbor, PointComparer> point_neighbors_type;
typedef unordered_map<int, point_neighbors_type> neighbors_container_type;

class AbstractAllKnnAlgorithm
{
    public:
        virtual ~AbstractAllKnnAlgorithm();
        virtual AllKnnResult* Process(const AllKnnProblem& problem) const = 0;

    protected:
        AbstractAllKnnAlgorithm();

        /** \brief Creates the container of nearest neighbors for all points
         *
         * \param numPoints The number of points
         * \param numNeighbors The number of nearest neighbors
         * \return An unordered map of point Ids to multisets of Neighbors
         *
         */

        neighbors_container_type* CreateNeighborsContainer(int numPoints, int numNeighbors) const;
};

#endif // ABSTRACTALLKNNALGORITHM_H
