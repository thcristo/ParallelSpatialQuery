/* Parallel plane sweep algorithm implementation using OpenMP
    This implementation operates on a copy of the original datasets and it is based on PlaneSweepCopyAlgorithm
 */
#ifndef PLANESWEEPCOPYPARALLELALGORITHM_H
#define PLANESWEEPCOPYPARALLELALGORITHM_H

#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "AbstractAllKnnAlgorithm.h"
#include <chrono>
#include <omp.h>

/** \brief Parallel plane sweep algorithm using copy of original datasets and OpenMP
 */
class PlaneSweepCopyParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepCopyParallelAlgorithm(int numThreads, bool parallelSort) : numThreads(numThreads), parallelSort(parallelSort)
        {
        }

        virtual ~PlaneSweepCopyParallelAlgorithm() {}

        string GetTitle() const
        {
            return parallelSort ? "Plane sweep copy parallel (parallel sorting)" : "Plane sweep copy parallel";
        }

        string GetPrefix() const
        {
            return parallelSort ? "planesweep_copy_parallel_psort" : "planesweep_copy_parallel";
        }

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            //the implementation is similar to PlaneSweepCopyAlgorithm
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
            }

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<AllKnnResultSorted>(new AllKnnResultSorted(problem, GetPrefix(), parallelSort));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDataset = pResult->GetTrainingDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            //OpenMP parallel loop through all input points
            #pragma omp parallel for schedule(dynamic)
            for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
            {
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                //in the parallel algorithm we have to do a binary search to find the next training point
                //this is in contrast to the serial version of the algorithm where we can use the value from the previous repetition of the loop
                auto nextTrainingPointIter = lower_bound(trainingDatasetBegin, trainingDatasetEnd, inputPointIter->x,
                                    [&](const Point& point, const double& value) { return point.x < value; } );

                auto prevTrainingPointIter = nextTrainingPointIter;
                if (prevTrainingPointIter > trainingDatasetBegin)
                {
                    --prevTrainingPointIter;
                }

                bool lowStop = prevTrainingPointIter == nextTrainingPointIter;
                bool highStop = nextTrainingPointIter == trainingDatasetEnd;

                //start moving left and right of the input point in x axis
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

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            pResult->setDuration(elapsed);
            pResult->setDurationSorting(elapsedSorting);
            pResult->setNeighborsContainer(pNeighborsContainer);

            return pResult;
        }

    private:
        int numThreads = 0;
        bool parallelSort = false;
};

#endif // PLANESWEEPCOPYPARALLELALGORITHM_H
