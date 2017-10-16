#include "ParallelPlaneSweepAlgorithm.h"

ParallelPlaneSweepAlgorithm::ParallelPlaneSweepAlgorithm() : AbstractAllKnnAlgorithm()
{
    //ctor
}

ParallelPlaneSweepAlgorithm::~ParallelPlaneSweepAlgorithm()
{
    //dtor
}

AllKnnResult* ParallelPlaneSweepAlgorithm::Process(const AllKnnProblem& problem) const
{
    return new AllKnnResult();
}
