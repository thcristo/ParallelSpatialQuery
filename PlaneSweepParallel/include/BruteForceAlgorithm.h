/* Brute  force algorithm implementation */

#ifndef BRUTEFORCEALGORITHM_H
#define BRUTEFORCEALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <chrono>

/** \brief Brute force algorithm
 */
class BruteForceAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceAlgorithm() {}

        virtual ~BruteForceAlgorithm() {}

        std::string GetTitle() const
        {
            return "Brute force";
        }

        std::string GetPrefix() const
        {
            return "bruteforce_serial";
        }

        std::unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            int numNeighbors = problem.GetNumNeighbors();

            //allocate vector for neighbors
            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            auto start = std::chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            //loop through all input points
            for (auto inputPoint = inputDatasetBegin; inputPoint < inputDatasetEnd; ++inputPoint)
            {
                //get the neighbors of this input point
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
};

#endif // BRUTEFORCEALGORITHM_H
