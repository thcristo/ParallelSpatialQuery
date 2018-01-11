#ifndef PLANESWEEPCOPYPARALLELALGORITHM_H
#define PLANESWEEPCOPYPARALLELALGORITHM_H

#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "AbstractAllKnnAlgorithm.h"
#include <chrono>
#include <omp.h>

template<class ProblemT, class ResultT, class ResultBaseT, class PointVectorT, class PointVectorIteratorT, class NeighborVectorT>
class PlaneSweepCopyParallelAlgorithm : public AbstractAllKnnAlgorithm<ProblemT, ResultBaseT, PointVectorT, PointVectorIteratorT, NeighborVectorT>
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

        unique_ptr<ResultBaseT> Process(ProblemT& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
            }

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<ResultT>(new ResultT(problem, GetPrefix(), parallelSort));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDataset = pResult->GetTrainingDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            //auto inputDatasetSize = inputDataset.size();

            //#pragma omp parallel
            //{
                /*
                int iThread = omp_get_thread_num();
                int numThreads = omp_get_num_threads();
                auto partitionSize = inputDatasetSize/numThreads;

                auto partitionStart = inputDatasetBegin + iThread*partitionSize;
                auto startSearchPos = lower_bound(trainingDatasetBegin, trainingDatasetEnd, partitionStart->x,
                                        [&](const Point& point, const double& value) { return point.x < value; } );
                */
                #pragma omp parallel for schedule(dynamic)
                for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
                {
                    auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);


                    auto nextTrainingPointIter = lower_bound(trainingDatasetBegin, trainingDatasetEnd, inputPointIter->x,
                                        [&](const Point& point, const double& value) { return point.x < value; } );


                    /*
                    auto nextTrainingPointIter = startSearchPos;
                    while (nextTrainingPointIter < trainingDatasetEnd && nextTrainingPointIter->x < inputPointIter->x)
                    {
                        ++nextTrainingPointIter;
                    }

                    startSearchPos = nextTrainingPointIter;
                    */
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
                            if (this->CheckAddNeighbor(inputPointIter, prevTrainingPointIter, neighbors))
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
                            if (this->CheckAddNeighbor(inputPointIter, nextTrainingPointIter, neighbors))
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
            //}


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
