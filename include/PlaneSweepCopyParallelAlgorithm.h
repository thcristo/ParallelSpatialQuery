#ifndef PLANESWEEPCOPYPARALLELALGORITHM_H
#define PLANESWEEPCOPYPARALLELALGORITHM_H

#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "AbstractAllKnnAlgorithm.h"
#include <chrono>
#include <omp.h>

class PlaneSweepCopyParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepCopyParallelAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }

        virtual ~PlaneSweepCopyParallelAlgorithm() {}

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<AllKnnResultSorted>(new AllKnnResultSorted(problem, "planesweep_copy_parallel"));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDataset = pResult->GetTrainingDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            auto inputDatasetSize = inputDataset.size();

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
            }

            #pragma omp parallel
            {
                int iThread = omp_get_thread_num();
                int numThreads = omp_get_num_threads();
                auto partitionSize = inputDatasetSize/numThreads;

                auto partitionStart = inputDatasetBegin + iThread*partitionSize;
                auto startSearchPos = lower_bound(trainingDatasetBegin, trainingDatasetEnd, partitionStart->x,
                                        [&](const Point& point, const double& value) { return point.x < value; } );

                #pragma omp for
                for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
                {
                    auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                    /*
                    auto nextTrainingPointIter = lower_bound(trainingDatasetBegin, trainingDatasetEnd, inputPointIter->x,
                                        [&](const Point& point, const double& value) { return point.x < value; } );
                    */

                    auto nextTrainingPointIter = startSearchPos;
                    while (nextTrainingPointIter < trainingDatasetEnd && nextTrainingPointIter->x < inputPointIter->x)
                    {
                        ++nextTrainingPointIter;
                    }

                    startSearchPos = nextTrainingPointIter;

                    auto prevTrainingPointIter = nextTrainingPointIter;
                    if (prevTrainingPointIter > trainingDatasetBegin)
                    {
                        --prevTrainingPointIter;
                    }

                    bool lowStop = prevTrainingPointIter == nextTrainingPointIter;
                    bool highStop = nextTrainingPointIter == trainingDatasetEnd;

                    while (!lowStop || !highStop)
                    {
                        if (!lowStop)
                        {
                            if (CheckAddNeighbor(inputPointIter, prevTrainingPointIter, neighbors))
                            {
                                if (prevTrainingPointIter > trainingDatasetBegin)
                                {
                                    --prevTrainingPointIter;
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
                            if (CheckAddNeighbor(inputPointIter, nextTrainingPointIter, neighbors))
                            {
                                if (nextTrainingPointIter < trainingDatasetEnd)
                                {
                                    ++nextTrainingPointIter;
                                }

                                if (nextTrainingPointIter == trainingDatasetEnd)
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
            }


            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            pResult->setDuration(elapsed);
            pResult->setDurationSorting(elapsedSorting);
            pResult->setNeighborsContainer(pNeighborsContainer);

            return pResult;
        }

    private:
        int numThreads;
};

#endif // PLANESWEEPCOPYPARALLELALGORITHM_H
