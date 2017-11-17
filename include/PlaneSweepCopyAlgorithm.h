#ifndef PLANESWEEPCOPYALGORITHM_H
#define PLANESWEEPCOPYALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>
#include <cmath>

class PlaneSweepCopyAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepCopyAlgorithm() {}
        virtual ~PlaneSweepCopyAlgorithm() {}

        unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) const override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_priority_queue_container_t> pNeighborsContainer =
                this->CreateNeighborsContainer<neighbors_priority_queue_t>(problem.GetInputDataset(), numNeighbors);

            auto start = chrono::high_resolution_clock::now();

            auto& inputDataset = problem.GetInputDatasetSorted();
            auto& trainingDataset = problem.GetTrainingDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto startSearchPos = trainingDataset.cbegin();

            for (auto inputPointIter = inputDataset.cbegin(); inputPointIter != inputDataset.cend(); ++inputPointIter)
            {
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id);

                /*
                auto nextTrainingPointIter = lower_bound(startSearchPos, trainingDataset.cend(), inputPointIter->x,
                                    [&](const Point& point, double& value) { return point.x < value; } );
                */

                auto nextTrainingPointIter = startSearchPos;
                while (nextTrainingPointIter < trainingDataset.cend() && nextTrainingPointIter->x < inputPointIter->x)
                {
                    ++nextTrainingPointIter;
                }

                startSearchPos = nextTrainingPointIter;
                auto prevTrainingPointIter = nextTrainingPointIter;
                if (prevTrainingPointIter > trainingDataset.cbegin())
                {
                    --prevTrainingPointIter;
                }

                bool lowStop = prevTrainingPointIter == nextTrainingPointIter;
                bool highStop = nextTrainingPointIter == trainingDataset.cend();

                while (!lowStop || !highStop)
                {
                    if (!lowStop)
                    {
                        if (CheckAddNeighbor(inputPointIter, prevTrainingPointIter, neighbors))
                        {
                            if (prevTrainingPointIter > trainingDataset.cbegin())
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
                            if (nextTrainingPointIter < trainingDataset.cend())
                            {
                                ++nextTrainingPointIter;
                            }

                            if (nextTrainingPointIter == trainingDataset.cend())
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

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, elapsedSorting, "planesweep_copy_serial", problem));
        }

    private:
};

#endif // PLANESWEEPCOPYALGORITHM_H
