#ifndef PLANESWEEPALGORITHM_H
#define PLANESWEEPALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <cmath>

template<class ProblemT, class ResultT, class PointVectorT, class PointVectorIteratorT, class NeighborVectorT, class PointVectorIndexT>
class PlaneSweepAlgorithm : public AbstractAllKnnAlgorithm<ProblemT, ResultT, PointVectorT, PointVectorIteratorT, NeighborVectorT>
{
    public:
        PlaneSweepAlgorithm() {}
        virtual ~PlaneSweepAlgorithm() {}

        string GetTitle() const
        {
            return "Plane sweep";
        }

        string GetPrefix() const
        {
            return "planesweep_serial";
        }

        unique_ptr<ResultT> Process(ProblemT& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            auto pNeighborsContainer =
                this->CreateNeighborsContainer(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            auto start = chrono::high_resolution_clock::now();

            PointVectorIndexT inputDatasetIndex(inputDataset.size());
            PointVectorIndexT trainingDatasetIndex(trainingDataset.size());

            PointVectorIteratorT m = inputDataset.cbegin();
            PointVectorIteratorT n = trainingDataset.cbegin();

            generate(inputDatasetIndex.begin(), inputDatasetIndex.end(), [&m] { return m++; } );
            generate(trainingDatasetIndex.begin(), trainingDatasetIndex.end(), [&n] { return n++; } );

            sort(inputDatasetIndex.begin(), inputDatasetIndex.end(),
                 [&](const PointVectorIteratorT& iter1, const PointVectorIteratorT& iter2)
                 {
                     return iter1->x < iter2->x;
                 });

            sort(trainingDatasetIndex.begin(), trainingDatasetIndex.end(),
                 [&](const PointVectorIteratorT& iter1, const PointVectorIteratorT& iter2)
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

                auto nextTrainingPointIndex = startSearchPos;
                while (nextTrainingPointIndex < trainingDatasetIndexEnd && (*nextTrainingPointIndex)->x < inputPointIter->x)
                {
                    ++nextTrainingPointIndex;
                }

                startSearchPos = nextTrainingPointIndex;
                auto prevTrainingPointIndex = nextTrainingPointIndex;
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
                        if (this->CheckAddNeighbor(inputPointIter, *prevTrainingPointIndex, neighbors))
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
                        if (this->CheckAddNeighbor(inputPointIter, *nextTrainingPointIndex, neighbors))
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

            return unique_ptr<ResultT>(new ResultT(problem, GetPrefix(), pNeighborsContainer, elapsed, elapsedSorting));
        }

    private:

};

#endif // PLANESWEEPALGORITHM_H
