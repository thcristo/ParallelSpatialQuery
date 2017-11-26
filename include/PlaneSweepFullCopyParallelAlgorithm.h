#ifndef PLANESWEEPFULLCOPYPARALLELALGORITHM_H
#define PLANESWEEPFULLCOPYPARALLELALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>


class PlaneSweepFullCopyParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepFullCopyParallelAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }

        virtual ~PlaneSweepFullCopyParallelAlgorithm() {}

                unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto start = chrono::high_resolution_clock::now();

            auto& inputDataset = problem.GetInputDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto& trainingDatasetCopy = problem.GetTrainingDatasetSortedCopy();

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
            int numThreads = 0;

            if (numThreads > 0)
            {
                omp_set_num_threads(numThreads);
            }

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

            stringstream ss;
            ss << "planesweep_full_copy_parallel_" << numThreads;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, elapsedSorting, ss.str(), problem));
        }
    protected:

    private:
        int numThreads;
};

#endif // PLANESWEEPFULLCOPYPARALLELALGORITHM_H
