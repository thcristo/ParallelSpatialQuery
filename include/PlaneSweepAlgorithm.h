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
            size_t numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_priority_queue_container_t> pNeighborsContainer =
                this->CreateNeighborsContainer<neighbors_priority_queue_t>(problem.GetInputDataset(), numNeighbors);


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

            auto finishSorting = chrono::high_resolution_clock::now();

            point_vector_index_iterator_t startSearchPos = trainingDatasetIndex.cbegin();
            const point_vector_index_iterator_t& trainingDatasetIndexBegin = trainingDatasetIndex.cbegin();
            const point_vector_index_iterator_t& trainingDatasetIndexEnd = trainingDatasetIndex.cend();

            for (auto inputPointIndex = inputDatasetIndex.cbegin(); inputPointIndex != inputDatasetIndex.cend(); ++inputPointIndex)
            {
                auto& inputPointIter = *inputPointIndex;
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id);


                /*
                point_vector_index_iterator_t nextTrainingPointIndex = lower_bound(startSearchPos, trainingDatasetIndex.cend(), inputPointIter->x,
                                    [&](const point_vector_iterator_t& iter, const double& value) { return iter->x < value; } );
                */

                point_vector_index_iterator_t nextTrainingPointIndex = startSearchPos;
                while (nextTrainingPointIndex < trainingDatasetIndexEnd && (*nextTrainingPointIndex)->x < inputPointIter->x)
                {
                    ++nextTrainingPointIndex;
                }

                startSearchPos = nextTrainingPointIndex;
                point_vector_index_iterator_t prevTrainingPointIndex = nextTrainingPointIndex;
                if (prevTrainingPointIndex > trainingDatasetIndexBegin)
                {
                    --prevTrainingPointIndex;
                }

                bool lowStop = prevTrainingPointIndex == nextTrainingPointIndex;
                bool highStop = nextTrainingPointIndex == trainingDatasetIndexEnd;

                while (!lowStop || !highStop)
                {
                    if (!lowStop)
                    {
                        if (CheckAddNeighbor(inputPointIter, *prevTrainingPointIndex, neighbors))
                        {
                            if (prevTrainingPointIndex > trainingDatasetIndexBegin)
                            {
                                --prevTrainingPointIndex;
                            }
                            else
                            {
                                lowStop = true;
                            }
                        }
                        else
                        {
                            lowStop = true;
                        }
                    }

                    if (!highStop)
                    {
                        if (CheckAddNeighbor(inputPointIter, *nextTrainingPointIndex, neighbors))
                        {
                            if (nextTrainingPointIndex < trainingDatasetIndexEnd)
                            {
                                ++nextTrainingPointIndex;
                            }

                            if (nextTrainingPointIndex == trainingDatasetIndexEnd)
                            {
                                highStop = true;
                            }
                        }
                        else
                        {
                            highStop = true;
                        }
                    }
                }
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, elapsedSorting, "planesweep_serial", problem));
        }

    private:
};

#endif // PLANESWEEPALGORITHM_H
