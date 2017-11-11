#ifndef BRUTEFORCEALGORITHM_H
#define BRUTEFORCEALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>
#include <chrono>

class BruteForceAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        BruteForceAlgorithm() {}

        virtual ~BruteForceAlgorithm() {}


        unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const override
        {
            int numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_priority_queue_container_t> pNeighborsContainer =
                this->CreateNeighborsContainer<neighbors_priority_queue_t>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            auto start = chrono::high_resolution_clock::now();

            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                auto& neighbors = pNeighborsContainer->at(inputPoint->id);

                for (auto trainingPoint = trainingDataset.cbegin(); trainingPoint != trainingDataset.cend(); ++trainingPoint)
                {
                    AddNeighbor(inputPoint, trainingPoint, neighbors);
                }
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, "bruteforce_serial", problem));
        }

    private:
};

#endif // BRUTEFORCEALGORITHM_H
