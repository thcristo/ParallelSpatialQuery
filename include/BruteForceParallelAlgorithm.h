#ifndef BRUTEFORCEPARALLELALGORITHM_H
#define BRUTEFORCEPARALLELALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <omp.h>

class BruteForceParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceParallelAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }

        virtual ~BruteForceParallelAlgorithm() {}

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
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

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
            }

            #pragma omp parallel for
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

            return unique_ptr<AllKnnResult>(new AllKnnResult(problem, "bruteforce_parallel", pNeighborsContainer, elapsed, chrono::duration<double>()));
        }

    private:
        int numThreads;
};

#endif // BRUTEFORCEPARALLELALGORITHM_H
