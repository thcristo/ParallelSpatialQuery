#ifndef BRUTEFORCEALGORITHM_H
#define BRUTEFORCEALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <chrono>

class BruteForceAlgorithm : public AbstractAllKnnAlgorithm
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

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            int numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

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
                    AddNeighbor(inputPoint, trainingPoint, neighbors);
                }
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(problem, GetPrefix(), pNeighborsContainer, elapsed, chrono::duration<double>()));
        }

    private:
};

#endif // BRUTEFORCEALGORITHM_H
