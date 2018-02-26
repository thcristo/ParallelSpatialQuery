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

struct Neighbor {
    const Point* point;
    double distanceSquared;
};



class NeighborComparer
{
    public :
        bool operator()(const Neighbor& n1, const Neighbor& n2) const
        {
            return n1.distanceSquared < n2.distanceSquared;
        }
};


bool endsWith(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

typedef vector<Neighbor> neighbors_vector_t;
typedef deque<Neighbor> neighbors_deque_t;
typedef priority_queue<Neighbor, neighbors_vector_t, NeighborComparer> neighbors_priority_queue_t;
typedef vector<Point> point_vector_t;
typedef vector<point_vector_t> point_vector_vector_t;
typedef point_vector_t::const_iterator point_vector_iterator_t;
typedef vector<point_vector_iterator_t> point_vector_index_t;
typedef point_vector_index_t::const_iterator point_vector_index_iterator_t;

template <class charT, charT decimalSeparator, charT thousandsSeparator>
class punct_facet: public numpunct<charT> {
protected:
    charT do_decimal_point() const { return decimalSeparator; }
    charT do_thousands_sep() const { return thousandsSeparator; }
    string do_grouping() const { return "\03"; }
};

#endif // PLANESWEEPPARALLEL_H_INCLUDED
