#ifndef PLANESWEEPPARALLEL_H_INCLUDED
#define PLANESWEEPPARALLEL_H_INCLUDED


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


typedef priority_queue<Neighbor, vector<Neighbor>, NeighborComparer> neighbors_priority_queue_t;
typedef vector<Point> point_vector_t;
typedef point_vector_t::const_iterator point_vector_iterator_t;
typedef vector<point_vector_iterator_t> point_vector_index_t;
typedef point_vector_index_t::const_iterator point_vector_index_iterator_t;

#endif // PLANESWEEPPARALLEL_H_INCLUDED
