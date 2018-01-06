#ifndef PLANESWEEPCOPYALGORITHM_H
#define PLANESWEEPCOPYALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <cmath>

template<class ProblemT, class ResultT, class ResultBaseT, class PointVectorT, class PointVectorIteratorT, class NeighborsContainerT>
class PlaneSweepCopyAlgorithm : public AbstractAllKnnAlgorithm<ProblemT, ResultBaseT, PointVectorT, PointVectorIteratorT>
{
    public:
        PlaneSweepCopyAlgorithm() {}
        virtual ~PlaneSweepCopyAlgorithm() {}

        string GetTitle() const
        {
            return "Plane sweep copy";
        }

        string GetPrefix() const
        {
            return "planesweep_copy_serial";
        }

        unique_ptr<ResultBaseT> Process(ProblemT& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->template CreateNeighborsContainer<NeighborsContainerT>(problem.GetInputDataset(), numNeighbors);

            auto start = chrono::high_resolution_clock::now();

            auto pResult = unique_ptr<ResultT>(new ResultT(problem, GetPrefix()));

            auto& inputDataset = pResult->GetInputDatasetSorted();
            auto& trainingDataset = pResult->GetTrainingDatasetSorted();

            auto finishSorting = chrono::high_resolution_clock::now();

            auto trainingDatasetBegin = trainingDataset.cbegin();
            auto trainingDatasetEnd = trainingDataset.cend();
            auto inputDatasetBegin = inputDataset.cbegin();
            auto inputDatasetEnd = inputDataset.cend();

            auto startSearchPos = trainingDatasetBegin;

            for (auto inputPointIter = inputDatasetBegin; inputPointIter < inputDatasetEnd; ++inputPointIter)
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

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;
            chrono::duration<double> elapsedSorting = finishSorting - start;

            pResult->setDuration(elapsed);
            pResult->setDurationSorting(elapsedSorting);
            pResult->setNeighborsContainer(pNeighborsContainer);

            return pResult;
        }

    private:
};

#endif // PLANESWEEPCOPYALGORITHM_H
