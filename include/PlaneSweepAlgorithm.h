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

            point_vector_index_t trainingDatasetIndex(trainingDataset.size());

            point_vector_iterator_t n = trainingDataset.cbegin();

            generate(trainingDatasetIndex.begin(), trainingDatasetIndex.end(), [&n] { return n++; } );

            sort(trainingDatasetIndex.begin(), trainingDatasetIndex.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->x < iter2->x;
                 });

            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                auto& neighbors = pNeighborsContainer->at(inputPoint->id);

                point_vector_index_iterator_t nextTrainingPointIndex = lower_bound(trainingDatasetIndex.cbegin(), trainingDatasetIndex.cend(), inputPoint->x,
                                    [&](const point_vector_iterator_t& iter, const double& value) { return iter->x < value; } );

                point_vector_index_iterator_t prevTrainingPointIndex = nextTrainingPointIndex;
                if (prevTrainingPointIndex > trainingDatasetIndex.cbegin())
                {
                    --prevTrainingPointIndex;
                }

                bool lowStop = prevTrainingPointIndex == nextTrainingPointIndex;
                bool highStop = nextTrainingPointIndex == trainingDatasetIndex.cend();

                while (!lowStop || !highStop)
                {
                    if (!lowStop)
                    {
                        double dx = inputPoint->x - (*prevTrainingPointIndex)->x;
                        double dxSquared = dx*dx;
                        double maxDistance = neighbors.top().distanceSquared;

                        if (dxSquared > maxDistance)
                        {
                            lowStop = true;
                        }
                        else
                        {
                            CheckInsertNeighbor(inputPoint, *prevTrainingPointIndex, neighbors);

                            if (prevTrainingPointIndex > trainingDatasetIndex.cbegin())
                            {
                                --prevTrainingPointIndex;
                            }
                            else
                            {
                                lowStop = true;
                            }
                        }
                    }

                    if (!highStop)
                    {
                        double dx = (*nextTrainingPointIndex)->x - inputPoint->x;
                        double dxSquared = dx*dx;
                        double maxDistance = neighbors.top().distanceSquared;

                        if (dxSquared > maxDistance)
                        {
                            highStop = true;
                        }
                        else
                        {
                            CheckInsertNeighbor(inputPoint, *nextTrainingPointIndex, neighbors);

                            if (nextTrainingPointIndex < trainingDatasetIndex.cend())
                            {
                                ++nextTrainingPointIndex;
                            }

                            if (nextTrainingPointIndex == trainingDatasetIndex.cend())
                            {
                                highStop = true;
                            }
                        }
                    }
                }
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, "planesweep_serial", problem));
        }
    protected:

    private:
};

#endif // PLANESWEEPALGORITHM_H
