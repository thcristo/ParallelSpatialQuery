#ifndef PLANESWEEPALGORITHM_H
#define PLANESWEEPALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>


class PlaneSweepAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepAlgorithm() {}
        virtual ~PlaneSweepAlgorithm() {}

        unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const override
        {
            int numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_container_t> pNeighborsContainer = this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            auto start = chrono::high_resolution_clock::now();

            vector<point_vector_size_t> trainingDatasetIndex(trainingDataset.size());

            point_vector_size_t n = 0;

            generate(trainingDatasetIndex.begin(), trainingDatasetIndex.end(), [&n] { return n++; } );

            sort(trainingDatasetIndex.begin(), trainingDatasetIndex.end(),
                 [&](const point_vector_size_t& index1, const point_vector_size_t& index2)
                 {
                     return trainingDataset[index1].x < trainingDataset[index2].x;
                 });

            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                auto& neighbors = pNeighborsContainer->at(inputPoint->id);

                int neighborsFound = 0;

                point_vector_index_t nextTrainingPoint = lower_bound(trainingDatasetIndex.begin(), trainingDatasetIndex.end(), inputPoint->x,
                                    [&](const point_vector_size_t& index, const double& value) { return trainingDataset[index].x < value; } );

                do
                {

                } while (neighborsFound < numNeighbors);
                /*
                for (auto trainingPoint = trainingDataset.cbegin(); trainingPoint != trainingDataset.cend(); ++trainingPoint)
                {
                    CheckInsertNeighbor(inputPoint, trainingPoint, neighbors);
                }
                */
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, "bruteforce_serial", problem));
        }
    protected:

    private:
};

#endif // PLANESWEEPALGORITHM_H
