#ifndef PLANESWEEPPARALLEL_H_INCLUDED
#define PLANESWEEPPARALLEL_H_INCLUDED


#include <queue>
#include <vector>
#include <deque>
#include <fstream>
#include <stxxl.h>

#define EXT_PAGE_SIZE  4
#define EXT_CACHE_PAGES  8
#define EXT_BLOCK_SIZE 2*1024*1024
#define EXT_ALLOC_STRATEGY stxxl::RC
#define EXT_PAGER_TYPE stxxl::pager_type::lru

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

template<typename T>
using ext_vector = typename stxxl::VECTOR_GENERATOR<T, EXT_PAGE_SIZE, EXT_CACHE_PAGES, EXT_BLOCK_SIZE, EXT_ALLOC_STRATEGY, EXT_PAGER_TYPE>::result;


typedef ext_vector<Point> point_vector_ext_t;
typedef vector<point_vector_ext_t> point_vector_vector_ext_t;
typedef point_vector_ext_t::const_iterator point_vector_iterator_ext_t;
typedef ext_vector<point_vector_iterator_ext_t> point_vector_index_ext_t;
typedef point_vector_index_ext_t::const_iterator point_vector_index_iterator_ext_t;

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

typedef Neighbor<point_vector_iterator_ext_t> NeighborExt;
typedef NeighborComparer<point_vector_iterator_ext_t> NeighborComparerExt;

template<class PointVectorIteratorT>
using neighbors_vector_t = vector<Neighbor<PointVectorIteratorT>>;
//typedef deque<Neighbor<point_vector_iterator_t>>> neighbors_deque_t;
template<class PointVectorIteratorT>
using neighbors_priority_queue_t = priority_queue<Neighbor<PointVectorIteratorT>, neighbors_vector_t<PointVectorIteratorT>, NeighborComparer<PointVectorIteratorT>>;





#endif // PLANESWEEPPARALLEL_H_INCLUDED
