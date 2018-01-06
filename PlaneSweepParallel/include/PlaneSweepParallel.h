#ifndef PLANESWEEPPARALLEL_H_INCLUDED
#define PLANESWEEPPARALLEL_H_INCLUDED


#include <queue>
#include <vector>
#include <deque>
#include <fstream>

using namespace std;

struct Point {
    long id;
    double x;
    double y;
};

typedef vector<Point> point_vector_t;
typedef vector<point_vector_t> point_vector_vector_t;
typedef point_vector_t::const_iterator point_vector_iterator_t;
typedef vector<point_vector_iterator_t> point_vector_index_t;
typedef point_vector_index_t::const_iterator point_vector_index_iterator_t;

template<class PointVectorIteratorT>
struct Neighbor {
    PointVectorIteratorT point;
    double distanceSquared;
};

template<class PointVectorIteratorT>
class NeighborComparer
{
    public :
        bool operator()(const Neighbor<PointVectorIteratorT>& n1, const Neighbor<PointVectorIteratorT>& n2) const
        {
            return n1.distanceSquared < n2.distanceSquared;
        }
};

typedef Neighbor<point_vector_iterator_t> NeighborMem;
typedef NeighborComparer<point_vector_iterator_t> NeighborComparerMem;

template<class PointVectorIteratorT>
using neighbors_vector_t = vector<Neighbor<PointVectorIteratorT>>;
//typedef deque<Neighbor<point_vector_iterator_t>>> neighbors_deque_t;
template<class PointVectorIteratorT>
using neighbors_priority_queue_t = priority_queue<Neighbor<PointVectorIteratorT>, neighbors_vector_t<PointVectorIteratorT>, NeighborComparer<PointVectorIteratorT>>;





#endif // PLANESWEEPPARALLEL_H_INCLUDED
