#include "AbstractAllKnnAlgorithm.h"
#include <vector>
#include <limits>

AbstractAllKnnAlgorithm::AbstractAllKnnAlgorithm()
{
    //ctor
}

AbstractAllKnnAlgorithm::~AbstractAllKnnAlgorithm()
{
    //dtor
}


neighbors_container_type* AbstractAllKnnAlgorithm::CreateNeighborsContainer(int numPoints, int numNeighbors) const
{
    vector<Neighbor> defaultNeighbors(numNeighbors, {nullptr, numeric_limits<double>::max()});
    return new neighbors_container_type({{0, point_neighbors_type(defaultNeighbors.begin(), defaultNeighbors.end())}}, numPoints);
}

