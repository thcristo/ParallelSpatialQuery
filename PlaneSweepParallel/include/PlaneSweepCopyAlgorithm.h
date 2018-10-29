/* Serial plane sweep algorithm implementation
    This implementation operates on a copy of the original datasets so it is faster than the index version (PlaneSweepAlgorithm).
    The disadvantage is that it needs more memory
 */
#ifndef PLANESWEEPCOPYALGORITHM_H
#define PLANESWEEPCOPYALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <cmath>
#include "AllKnnResultSorted.h"

/** \brief Serial plane sweep algorithm using copy of original datasets
 */
class PlaneSweepCopyAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepCopyAlgorithm() {}
        virtual ~PlaneSweepCopyAlgorithm() {}

        std::string GetTitle() const
        {
            return "Plane sweep copy";
        }

        std::string GetPrefix() const
        {
            return "planesweep_copy_serial";
        }

        std::unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            //the implementation is the same as PlaneSweepAlgorithm with the only difference that it uses a copy of the original problem datasets
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto start = std::chrono::high_resolution_clock::now();

            auto pResult = std::unique_ptr<AllKnnResultSorted>(new AllKnnResultSorted(problem, GetPrefix()));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDataset = pResult->GetTrainingDatasetSorted();

            auto finishSorting = std::chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();


            auto startSearchPos = trainingDatasetBegin;

            for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
            {
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                auto nextTrainingPointIter = startSearchPos;
                while (nextTrainingPointIter < trainingDatasetEnd && nextTrainingPointIter->x < inputPointIter->x)
                {
                    ++nextTrainingPointIter;
                }

                //in the serial algorithm we can store the position of the next training point so we can use it in the next repetition
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

            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::chrono::duration<double> elapsedSorting = finishSorting - start;

            pResult->setDuration(elapsed);
            pResult->setDurationSorting(elapsedSorting);
            pResult->setNeighborsContainer(pNeighborsContainer);

            return pResult;
        }
};

#endif // PLANESWEEPCOPYALGORITHM_H
