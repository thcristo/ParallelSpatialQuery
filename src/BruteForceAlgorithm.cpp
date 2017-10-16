#include "BruteForceAlgorithm.h"


BruteForceAlgorithm::BruteForceAlgorithm()
{
    //ctor
}

BruteForceAlgorithm::~BruteForceAlgorithm()
{
    //dtor
}

AllKnnResult* BruteForceAlgorithm::Process(const AllKnnProblem& problem) const
{
    /*
    int inputPointsNum = problem.GetInputCount();
    int trainingPointsNum = problem.GetTrainingCount();
    int numNeighbors = problem.GetK();

    neighbors_container_type* pNeighbors = this->CreateNeighborsContainer(inputPointsNum, numNeighbors);

    for (int i=0; i < inputPointsNum; ++i)
    {
        for (int j=0; j < trainingPointsNum; ++j)
        {


        }
    }
    */
    return new AllKnnResult();
}
