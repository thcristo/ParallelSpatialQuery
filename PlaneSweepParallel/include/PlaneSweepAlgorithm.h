/* Serial plane sweep algorithm implementation
    This implementation operates on indexes of points so it has some performance overhead in comparison to the copy algorithm.
    The advantage is that it needs less memory
 */

#ifndef PLANESWEEPALGORITHM_H
#define PLANESWEEPALGORITHM_H

#include "AbstractAllKnnAlgorithm.h"
#include <cmath>

/** \brief Serial plane sweep algorithm using indexes to points
 */
class PlaneSweepAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        PlaneSweepAlgorithm() {}
        virtual ~PlaneSweepAlgorithm() {}

        std::string GetTitle() const
        {
            return "Plane sweep";
        }

        std::string GetPrefix() const
        {
            return "planesweep_serial";
        }

        std::unique_ptr<AllKnnResult> Process(AllKnnProblem& problem) override
        {
            size_t numNeighbors = problem.GetNumNeighbors();

            //allocate vector of neighbors for all input points
            auto pNeighborsContainer =
                this->CreateNeighborsContainer<pointNeighbors_priority_queue_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            auto start = std::chrono::high_resolution_clock::now();

            //copy the datasets
            point_vector_index_t inputDatasetIndex(inputDataset.size());
            point_vector_index_t trainingDatasetIndex(trainingDataset.size());

            point_vector_iterator_t m = inputDataset.cbegin();
            point_vector_iterator_t n = trainingDataset.cbegin();

            //fill vectors with indexes of points
            generate(inputDatasetIndex.begin(), inputDatasetIndex.end(), [&m] { return m++; } );
            generate(trainingDatasetIndex.begin(), trainingDatasetIndex.end(), [&n] { return n++; } );

            //sort the indexes
            sort(inputDatasetIndex.begin(), inputDatasetIndex.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->x < iter2->x;
                 });

            sort(trainingDatasetIndex.begin(), trainingDatasetIndex.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->x < iter2->x;
                 });

            auto finishSorting = std::chrono::high_resolution_clock::now();

            auto startSearchPos = trainingDatasetIndex.cbegin();
            auto trainingDatasetIndexBegin = trainingDatasetIndex.cbegin();
            auto trainingDatasetIndexEnd = trainingDatasetIndex.cend();
            auto inputDatasetIndexBegin = inputDatasetIndex.cbegin();
            auto inputDatasetIndexEnd = inputDatasetIndex.cend();

            //loop through all input points
            for (auto inputPointIndex = inputDatasetIndexBegin; inputPointIndex < inputDatasetIndexEnd; ++inputPointIndex)
            {
                auto& inputPointIter = *inputPointIndex;
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id - 1);

                //find the training point with x greater or equal to input point
                point_vector_index_iterator_t nextTrainingPointIndex = startSearchPos;
                while (nextTrainingPointIndex < trainingDatasetIndexEnd && (*nextTrainingPointIndex)->x < inputPointIter->x)
                {
                    ++nextTrainingPointIndex;
                }

                startSearchPos = nextTrainingPointIndex;
                //find the previous training point
                point_vector_index_iterator_t prevTrainingPointIndex = nextTrainingPointIndex;
                if (prevTrainingPointIndex > trainingDatasetIndexBegin)
                {
                    --prevTrainingPointIndex;
                }

                bool lowStop = prevTrainingPointIndex == nextTrainingPointIndex;
                bool highStop = nextTrainingPointIndex == trainingDatasetIndexEnd;

                //start moving left and right of the input point in x axis
                while (!lowStop || !highStop)
                {
                    //check if we can move toward lower x
                    if (!lowStop)
                    {
                        //check distance and add neighbor to heap
                        if (CheckAddNeighbor(inputPointIter, *prevTrainingPointIndex, neighbors))
                        {
                            if (prevTrainingPointIndex > trainingDatasetIndexBegin)
                            {
                                //move to previous training point
                                --prevTrainingPointIndex;
                            }
                            else
                            {
                                //we have reached the beginning of training dataset
                                lowStop = true;
                            }
                        }
                        else
                        {
                            //distance is greater than top of heap so stop examining points with a lower x
                            lowStop = true;
                        }
                    }

                    //check if we can move toward higher x
                    if (!highStop)
                    {
                         //check distance and add neighbor to heap
                        if (CheckAddNeighbor(inputPointIter, *nextTrainingPointIndex, neighbors))
                        {
                            if (nextTrainingPointIndex < trainingDatasetIndexEnd)
                            {
                                //move to next training point
                                ++nextTrainingPointIndex;
                            }

                            if (nextTrainingPointIndex == trainingDatasetIndexEnd)
                            {
                                //we have reached the end of training dataset
                                highStop = true;
                            }
                        }
                        else
                        {
                            //distance is greater than top of heap so stop examining points with a higher x
                            highStop = true;
                        }
                    }
                }
            }

            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::chrono::duration<double> elapsedSorting = finishSorting - start;

            return std::unique_ptr<AllKnnResult>(new AllKnnResult(problem, GetPrefix(), pNeighborsContainer, elapsed, elapsedSorting));
        }
};

#endif // PLANESWEEPALGORITHM_H
