#ifndef PLANESWEEPALGORITHM_H
#define PLANESWEEPALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <cmath>

class PlaneSweepAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepAlgorithm() {}
        virtual ~PlaneSweepAlgorithm() {}

        string GetTitle() const
        {
            return "Plane sweep";
        }

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

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

            auto startSearchPos = trainingDatasetIndex.cbegin();
            auto trainingDatasetIndexBegin = trainingDatasetIndex.cbegin();
            auto trainingDatasetIndexEnd = trainingDatasetIndex.cend();
            auto inputDatasetIndexBegin = inputDatasetIndex.cbegin();
            auto inputDatasetIndexEnd = inputDatasetIndex.cend();

            for (auto inputPointIndex = inputDatasetIndexBegin; inputPointIndex < inputDatasetIndexEnd; ++inputPointIndex)
            {
                auto& inputPointIter = *inputPointIndex;
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

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

            return unique_ptr<AllKnnResult>(new AllKnnResult(problem, "planesweep_serial", pNeighborsContainer, elapsed, elapsedSorting));
        }

    private:

};

#endif // PLANESWEEPALGORITHM_H
