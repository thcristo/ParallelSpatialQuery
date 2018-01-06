#ifndef BRUTEFORCEPARALLELALGORITHM_H
#define BRUTEFORCEPARALLELALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <omp.h>

template<class ProblemT, class ResultT, class PointVectorT, class PointVectorIteratorT, class NeighborsContainerT>
class BruteForceParallelAlgorithm : public AbstractAllKnnAlgorithm<ProblemT, ResultT, PointVectorT, PointVectorIteratorT>
{
    public:
        BruteForceParallelAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }

        virtual ~BruteForceParallelAlgorithm() {}

        string GetTitle() const
        {
            return "Brute force parallel";
        }

        string GetPrefix() const
        {
            return "bruteforce_parallel";
        }

        unique_ptr<ResultT> Process(ProblemT& problem) override
        {
            int numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->template CreateNeighborsContainer<NeighborsContainerT>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
            }

            auto start = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            #pragma omp parallel for schedule(dynamic)
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
        int numThreads = 0;
};

#endif // BRUTEFORCEPARALLELALGORITHM_H
