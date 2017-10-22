#ifndef PLANESWEEPPARALLEL_H_INCLUDED
#define PLANESWEEPPARALLEL_H_INCLUDED

#include <unordered_map>
#include <queue>
#include <vector>

typedef struct{
    long id;
    double x;
    double y;
} Point;

typedef struct{
    const Point* point;
    double distanceSquared;
} Neighbor;

class NeighborComparer
{
    public :
        bool operator()(const Neighbor& n1, const Neighbor& n2) const
        {
            return n1.distanceSquared < n2.distanceSquared;
        }
};


typedef priority_queue<Neighbor, vector<Neighbor>, NeighborComparer> point_neighbors_t;
typedef unordered_map<int, point_neighbors_t> neighbors_container_t;

#endif // PLANESWEEPPARALLEL_H_INCLUDED
