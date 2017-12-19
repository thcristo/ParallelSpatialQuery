#ifndef PLANESWEEPFULLCOPYPARALLELALGORITHM_H
#define PLANESWEEPFULLCOPYPARALLELALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"


class PlaneSweepFullCopyParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepFullCopyParallelAlgorithm(int numThreads, bool parallelSort) : numThreads(numThreads), parallelSort(parallelSort)
        {
        }

        string GetTitle() const
        {
            return parallelSort ? "Plane sweep full copy parallel (parallel sorting)" : "Plane sweep full copy parallel";
        }

        string GetPrefix() const
        {
            return parallelSort ? "planesweep_full_copy_parallel_psort" : "planesweep_full_copy_parallel";
        }

        virtual ~PlaneSweepFullCopyParallelAlgorithm() {}

                unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            int numThreads = 0;

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
            }

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<AllKnnResultSorted>(new AllKnnResultSorted(problem, GetPrefix(), parallelSort));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDatasetCopy = pResult->GetTrainingDatasetSortedCopy();

            auto finishSorting = chrono::high_resolution_clock::now();

            vector<point_vector_t::const_iterator> trainingDatasetBeginCopy;
            vector<point_vector_t::const_iterator> trainingDatasetEndCopy;

            for (auto copyIter = trainingDatasetCopy.cbegin(); copyIter < trainingDatasetCopy.cend(); ++copyIter)
            {
                trainingDatasetBeginCopy.push_back(copyIter->cbegin());
                trainingDatasetEndCopy.push_back(copyIter->cend());
            }

            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            auto inputDatasetSize = inputDataset.size();

            #pragma omp parallel shared(numThreads)
            {
                int iThread = omp_get_thread_num();
                numThreads = omp_get_num_threads();
                auto partitionSize = inputDatasetSize/numThreads;

                auto partitionStart = inputDatasetBegin + iThread*partitionSize;
                auto startSearchPos = lower_bound(trainingDatasetBeginCopy[iThread], trainingDatasetEndCopy[iThread], partitionStart->x,
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
                    while (nextTrainingPointIter < trainingDatasetEndCopy[iThread] && nextTrainingPointIter->x < inputPointIter->x)
                    {
                        ++nextTrainingPointIter;
                    }

                    startSearchPos = nextTrainingPointIter;

                    auto prevTrainingPointIter = nextTrainingPointIter;
                    if (prevTrainingPointIter > trainingDatasetBeginCopy[iThread])
                    {
                        --prevTrainingPointIter;
                    }

                    bool lowStop = prevTrainingPointIter == nextTrainingPointIter;
                    bool highStop = nextTrainingPointIter == trainingDatasetEndCopy[iThread];

                    while (!lowStop || !highStop)
                    {
                        if (!lowStop)
                        {
                            if (CheckAddNeighbor(inputPointIter, prevTrainingPointIter, neighbors))
                            {
                                if (prevTrainingPointIter > trainingDatasetBeginCopy[iThread])
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
                                if (nextTrainingPointIter < trainingDatasetEndCopy[iThread])
                                {
                                    ++nextTrainingPointIter;
                                }

                                if (nextTrainingPointIter == trainingDatasetEndCopy[iThread])
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
    protected:

    private:
        int numThreads;
        bool parallelSort;
};

#endif // PLANESWEEPFULLCOPYPARALLELALGORITHM_H
