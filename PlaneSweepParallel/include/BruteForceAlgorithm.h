#ifndef BRUTEFORCEALGORITHM_H
#define BRUTEFORCEALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <chrono>

template<class ProblemT, class ResultT, class PointVectorT, class PointVectorIteratorT, class NeighborVectorT>
class BruteForceAlgorithm : public AbstractAllKnnAlgorithm<ProblemT, ResultT, PointVectorT, PointVectorIteratorT, NeighborVectorT>
{
    public:
        BruteForceAlgorithm() {}

        virtual ~BruteForceAlgorithm() {}

        string GetTitle() const
        {
            return "Brute force";
        }

        string GetPrefix() const
        {
            return "bruteforce_serial";
        }

        unique_ptr<ResultT> Process(ProblemT& problem) override
        {
            int numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            auto start = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            for (auto inputPoint = inputDatasetBegin; inputPoint < inputDatasetEnd; ++inputPoint)
            {
                auto& neighbors = pNeighborsContainer->at(inputPoint->id - 1);

                for (auto trainingPoint = trainingDatasetBegin; trainingPoint < trainingDatasetEnd; ++trainingPoint)
                {
                    this->AddNeighbor(inputPoint, trainingPoint, neighbors);
                }
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<ResultT>(new ResultT(problem, GetPrefix(), pNeighborsContainer, elapsed, chrono::duration<double>()));
        }

    private:
};

#endif // BRUTEFORCEALGORITHM_H
