/* This file contains type and structure definitions used by all algorithms */
#ifndef PLANESWEEPPARALLEL_H_INCLUDED
#define PLANESWEEPPARALLEL_H_INCLUDED

#include <queue>
#include <vector>
#include <deque>
#include <fstream>
#include <stxxl/vector>

/** \brief Definition of point structure
 */
struct Point
{
    unsigned long id;
    double x;
    double y;
};

/** \brief Definition of neighbor
 */
struct Neighbor
{
    unsigned long pointId;
    double distanceSquared;
};

/** \brief Definition of neighbor for external memory algorithm
 *          We need also the input point id and rank of neighbor so we can sort the list of neighbors
 */
struct NeighborExt : public Neighbor
{
    unsigned long inputPointId;
    unsigned int position;
};

/** \brief Comparer for neighbors based on distance
 */
class NeighborComparer
{
    public :
        bool operator()(const Neighbor& n1, const Neighbor& n2) const
        {
            return n1.distanceSquared < n2.distanceSquared;
        }
};

/** \brief Point structure that keeps also the stripe where the point has been assigned to
 */
struct StripePoint : public Point
{
    size_t stripe;
};

bool endsWith(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

//Definitions of vector types
typedef std::vector<Neighbor> neighbors_vector_t;
typedef std::deque<Neighbor> neighbors_deque_t;
typedef std::priority_queue<Neighbor, neighbors_vector_t, NeighborComparer> neighbors_priority_queue_t;
typedef std::vector<Point> point_vector_t;
typedef std::vector<point_vector_t> point_vector_vector_t;
typedef point_vector_t::const_iterator point_vector_iterator_t;
typedef std::vector<point_vector_iterator_t> point_vector_index_t;
typedef point_vector_index_t::const_iterator point_vector_index_iterator_t;

//class used for output of numbers in greek format
template <class charT, charT decimalSeparator, charT thousandsSeparator>
class punct_facet: public std::numpunct<charT> {
protected:
    charT do_decimal_point() const { return decimalSeparator; }
    charT do_thousands_sep() const { return thousandsSeparator; }
    std::string do_grouping() const { return "\03"; }
};

//Definitions for external memory vectors
typedef stxxl::VECTOR_GENERATOR<Point>::result ext_point_vector_t;
typedef stxxl::VECTOR_GENERATOR<NeighborExt>::result ext_neighbors_vector_t;
typedef stxxl::VECTOR_GENERATOR<size_t>::result ext_size_vector_t;

#endif // PLANESWEEPPARALLEL_H_INCLUDED
