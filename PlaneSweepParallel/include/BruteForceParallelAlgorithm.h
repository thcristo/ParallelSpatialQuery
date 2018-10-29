/* Parallel brute force algorithm implementation using OpenMP */
#ifndef BRUTEFORCEPARALLELALGORITHM_H
#define BRUTEFORCEPARALLELALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <omp.h>

/** \brief Parallel brute force algorithm using OpenMP
 */
class BruteForceParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceParallelAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }

        virtual ~BruteForceParallelAlgorithm() {}

        std::string GetTitle() const
        {
            return "Brute force parallel";
        }

        std::string GetPrefix() const
        {
            return "bruteforce_parallel";
        }

        std::unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            int numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            //if numThreads=0, let the system decide the number of threads based on number of cores
            if (numThreads > 0)
            {
                //set number of threads to use
                omp_set_num_threads(numThreads);
            }

            auto start = std::chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            //parallel loop through all input points
            #pragma omp parallel for schedule(dynamic)
            for (auto inputPoint = inputDatasetBegin; inputPoint < inputDatasetEnd; ++inputPoint)
            {
                auto& neighbors = pNeighborsContainer->at(inputPoint->id - 1);

                //loop through all training points
                for (auto trainingPoint = trainingDatasetBegin; trainingPoint < trainingDatasetEnd; ++trainingPoint)
                {
                    //check distance and add neighbor to max heap
                    AddNeighbor(inputPoint, trainingPoint, neighbors);
                }
            }

            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;

            return std::unique_ptr<AllKnnResult>(new AllKnnResult(problem, GetPrefix(), pNeighborsContainer, elapsed, std::chrono::duration<double>()));
        }

    private:
        int numThreads = 0;
};

#endif // BRUTEFORCEPARALLELALGORITHM_H
