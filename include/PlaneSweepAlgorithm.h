#ifndef PLANESWEEPALGORITHM_H
#define PLANESWEEPALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>
#include <cmath>

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

            point_vector_index_t inputDatasetIndex(inputDataset.size());
            point_vector_index_t trainingDatasetIndex(trainingDataset.size());

            point_vector_iterator_t m = inputDataset.cbegin();
            point_vector_iterator_t n = trainingDataset.cbegin();

            generate(inputDatasetIndex.begin(), inputDatasetIndex.end(), [&m] { return m++; } );
            generate(trainingDatasetIndex.begin(), trainingDatasetIndex.end(), [&n] { return n++; } );

            sort(inputDatasetIndex.begin(), inputDatasetIndex.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->x < iter2->x;
                 });

            sort(trainingDatasetIndex.begin(), trainingDatasetIndex.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->x < iter2->x;
                 });

            point_vector_index_iterator_t startSearchPos = trainingDatasetIndex.cbegin();

            for (auto inputPointIndex = inputDatasetIndex.cbegin(); inputPointIndex != inputDatasetIndex.cend(); ++inputPointIndex)
            {
                auto inputPointIter = *inputPointIndex;
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id);

                /*
                point_vector_index_iterator_t nextTrainingPointIndex = lower_bound(startSearchPos, trainingDatasetIndex.cend(), inputPointIter->x,
                                    [&](const point_vector_iterator_t& iter, const double& value) { return iter->x < value; } );
                */
                point_vector_index_iterator_t nextTrainingPointIndex = startSearchPos;
                while (nextTrainingPointIndex < trainingDatasetIndex.cend() && (*nextTrainingPointIndex)->x < inputPointIter->x)
                {
                    ++nextTrainingPointIndex;
                }
                startSearchPos = nextTrainingPointIndex;
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
                        auto prevTrainingPointIter = *prevTrainingPointIndex;
                        double dx = inputPointIter->x - prevTrainingPointIter->x;
                        double dxSquared = dx*dx;
                        auto& topNeighbor = neighbors.top();
                        double maxDistance = topNeighbor.distanceSquared;

                        if (dxSquared > maxDistance)
                        {
                            lowStop = true;
                        }
                        else
                        {
                            CheckInsertNeighbor(inputPointIter, prevTrainingPointIter, neighbors);

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
                        auto nextTrainingPointIter = *nextTrainingPointIndex;
                        double dx = nextTrainingPointIter->x - inputPointIter->x;
                        double dxSquared = dx*dx;
                        auto& topNeighbor = neighbors.top();
                        double maxDistance = topNeighbor.distanceSquared;

                        if (dxSquared > maxDistance)
                        {
                            highStop = true;
                        }
                        else
                        {
                            CheckInsertNeighbor(inputPointIter, nextTrainingPointIter, neighbors);

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
