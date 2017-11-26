#ifndef PLANESWEEPSTRIPSPARALLELALGORITHM_H
#define PLANESWEEPSTRIPSPARALLELALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>


class PlaneSweepStripsParallelAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepStripsParallelAlgorithm(int numThreads) : numThreads(numThreads)
        {
        }
        virtual ~PlaneSweepStripsParallelAlgorithm() {}

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto start = chrono::high_resolution_clock::now();

            auto stripData = problem.GetStripData(numThreads);

            auto finishSorting = chrono::high_resolution_clock::now();



            auto startSearchPos = trainingDatasetBegin;

            #pragma omp parallel for
            for (int iStrip = 0; iStrip < numThreads; ++iStrip)
            {
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                /*
                auto nextTrainingPointIter = lower_bound(startSearchPos, trainingDataset.cend(), inputPointIter->x,
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

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, elapsedSorting, "planesweep_strips_parallel", problem));
        }
    protected:

    private:
        int numThreads;
};

#endif // PLANESWEEPSTRIPSPARALLELALGORITHM_H
