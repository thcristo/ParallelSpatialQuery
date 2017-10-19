#ifndef ABSTRACTALLKNNALGORITHM_H
#define ABSTRACTALLKNNALGORITHM_H

#include <set>
#include <unordered_map>
#include <vector>
#include <limits>
#include "AllKnnProblem.h"
#include "AllKnnResult.h"

using namespace std;

typedef struct{
    Point* point;
    double distance;
} Neighbor;

class PointComparer
{
    public :
        bool operator()(const Neighbor& n1, const Neighbor& n2) const
        {
            return n1.distance < n2.distance;
        }
};


typedef multiset<Neighbor, PointComparer> point_neighbors_t;
typedef unordered_map<int, point_neighbors_t> neighbors_container_t;

class AbstractAllKnnAlgorithm
{
    public:
        virtual ~AbstractAllKnnAlgorithm() {}
        virtual AllKnnResult* Process(const AllKnnProblem& problem) const = 0;

    protected:
        AbstractAllKnnAlgorithm() {}

        /** \brief Creates the container of nearest neighbors for all points
         *
         * \param numPoints The number of points
         * \param numNeighbors The number of nearest neighbors
         * \return An unordered map of point Ids to multisets of Neighbors
         *
         */

        neighbors_container_t* CreateNeighborsContainer(const vector<Point>& inputDataset, int numNeighbors) const
        {
            try
            {
                neighbors_container_t* pContainer = new neighbors_container_t(inputDataset.size());

                vector<Neighbor> defaultNeighbors(numNeighbors, {nullptr, numeric_limits<double>::max()});

                for (auto point = inputDataset.cbegin(); point != inputDataset.cend(); ++point)
                {
                    pContainer->insert(make_pair(point->id, point_neighbors_t(defaultNeighbors.cbegin(), defaultNeighbors.cend())));
                }

                return pContainer;
            }
            catch(bad_alloc)
            {
                throw ApplicationException("Cannot allocate memory for neighbors container.");
            }

        }
};

#endif // ABSTRACTALLKNNALGORITHM_H
