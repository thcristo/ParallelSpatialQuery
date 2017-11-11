#ifndef SWITCHINGPLANESWEEPALGORITHM_H
#define SWITCHINGPLANESWEEPALGORITHM_H

#include <AbstractAllKnnAlgorithm.h>
#include <array>

class SwitchingPlaneSweepAlgorithm : public AbstractAllKnnAlgorithm
{
    public:
        SwitchingPlaneSweepAlgorithm() {}
        virtual ~SwitchingPlaneSweepAlgorithm() {}

        unique_ptr<AllKnnResult> Process(const AllKnnProblem& problem) const override
        {
            int numNeighbors = problem.GetNumNeighbors();

            unique_ptr<neighbors_vector_container_t> pNeighborsContainer =
                this->CreateNeighborsContainer<neighbors_vector_t>(problem.GetInputDataset(), numNeighbors);

            auto& inputDataset = problem.GetInputDataset();
            auto& trainingDataset = problem.GetTrainingDataset();

            auto start = chrono::high_resolution_clock::now();

            point_vector_index_t inputDatasetIndexX(inputDataset.size());
            point_vector_index_t trainingDatasetIndexX(trainingDataset.size());

            point_vector_iterator_t m = inputDataset.cbegin();
            point_vector_iterator_t n = trainingDataset.cbegin();

            generate(inputDatasetIndexX.begin(), inputDatasetIndexX.end(), [&m] { return m++; } );
            generate(trainingDatasetIndexX.begin(), trainingDatasetIndexX.end(), [&n] { return n++; } );

            point_vector_index_t inputDatasetIndexY(inputDataset.size());
            point_vector_index_t trainingDatasetIndexY(trainingDataset.size());

            m = inputDataset.cbegin();
            n = trainingDataset.cbegin();

            generate(inputDatasetIndexY.begin(), inputDatasetIndexY.end(), [&m] { return m++; } );
            generate(trainingDatasetIndexY.begin(), trainingDatasetIndexY.end(), [&n] { return n++; } );

            sort(inputDatasetIndexX.begin(), inputDatasetIndexX.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->x < iter2->x;
                 });

            sort(trainingDatasetIndexX.begin(), trainingDatasetIndexX.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->x < iter2->x;
                 });

            sort(inputDatasetIndexY.begin(), inputDatasetIndexY.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->y < iter2->y;
                 });

            sort(trainingDatasetIndexY.begin(), trainingDatasetIndexY.end(),
                 [&](const point_vector_iterator_t& iter1, const point_vector_iterator_t& iter2)
                 {
                     return iter1->y < iter2->y;
                 });

            point_vector_index_iterator_t startSearchPosX = trainingDatasetIndexX.cbegin();
            array<double, 4> distances;
            array<bool, 4> distanceCalculated;

            for (auto inputPointIndex = inputDatasetIndexX.cbegin(); inputPointIndex != inputDatasetIndexX.cend(); ++inputPointIndex)
            {
                auto inputPointIter = *inputPointIndex;
                auto& neighbors = pNeighborsContainer->at(inputPointIter->id);

                /*
                point_vector_index_iterator_t nextTrainingPointIndex = lower_bound(startSearchPos, trainingDatasetIndex.cend(), inputPointIter->x,
                                    [&](const point_vector_iterator_t& iter, const double& value) { return iter->x < value; } );
                */
                point_vector_index_iterator_t nextTrainingPointIndexX = startSearchPosX;
                while (nextTrainingPointIndexX < trainingDatasetIndexX.cend() && (*nextTrainingPointIndexX)->x < inputPointIter->x)
                {
                    ++nextTrainingPointIndexX;
                }

                startSearchPosX = nextTrainingPointIndexX;
                point_vector_index_iterator_t prevTrainingPointIndexX = nextTrainingPointIndexX;
                if (prevTrainingPointIndexX > trainingDatasetIndexX.cbegin())
                {
                    --prevTrainingPointIndexX;
                }

                point_vector_index_iterator_t nextTrainingPointIndexY = lower_bound(trainingDatasetIndexY.cbegin(), trainingDatasetIndexY.cend(), inputPointIter->y,
                                    [&](const point_vector_iterator_t& iter, const double& value) { return iter->y < value; } );

                point_vector_index_iterator_t prevTrainingPointIndexY = nextTrainingPointIndexY;
                if (prevTrainingPointIndexY > trainingDatasetIndexY.cbegin())
                {
                    --prevTrainingPointIndexY;
                }

                double maxDouble = numeric_limits<double>::max();
                distances.fill(maxDouble);
                distanceCalculated.fill(false);

                for (int iNeighbor = 0; iNeighbor < numNeighbors; ++iNeighbor)
                {
                    if (!distanceCalculated[0])
                    {
                        if (prevTrainingPointIndexX < nextTrainingPointIndexX)
                        {
                            distances[0] = CalcDistanceSquared(inputPointIter, *prevTrainingPointIndexX);
                        }
                        else
                        {
                            distances[0] = maxDouble;
                        }
                        distanceCalculated[0] = true;
                    }

                    if (!distanceCalculated[1])
                    {
                        if (nextTrainingPointIndexX < trainingDatasetIndexX.cend())
                        {
                            distances[1] = CalcDistanceSquared(inputPointIter, *nextTrainingPointIndexX);
                        }
                        else
                        {
                            distances[0] = maxDouble;
                        }
                        distanceCalculated[1] = true;
                    }

                    if (!distanceCalculated[2])
                    {
                        if (prevTrainingPointIndexY < nextTrainingPointIndexY)
                        {
                            distances[2] = CalcDistanceSquared(inputPointIter, *prevTrainingPointIndexY);
                        }
                        else
                        {
                            distances[2] = maxDouble;
                        }
                        distanceCalculated[2] = true;
                    }

                    if (!distanceCalculated[3])
                    {
                        if (nextTrainingPointIndexY < trainingDatasetIndexY.cend())
                        {
                            distances[3] = CalcDistanceSquared(inputPointIter, *nextTrainingPointIndexY);
                        }
                        else
                        {
                            distances[3] = maxDouble;
                        }
                        distanceCalculated[3] = true;
                    }

                    auto minDistanceIter = min_element(distances.cbegin(), distances.cend());
                    int minDistancePos = distance(distances.cbegin(), minDistanceIter);

                    switch(minDistancePos)
                    {
                        case 0:
                            AddNeighbor(*prevTrainingPointIndexX, *minDistanceIter, neighbors);
                            if (prevTrainingPointIndexX > trainingDatasetIndexX.cbegin())
                            {
                                --prevTrainingPointIndexX;
                                distanceCalculated[0] = false;
                            }
                            else
                            {
                                distances[0] = maxDouble;
                            }
                            break;
                        case 1:
                            AddNeighbor(*nextTrainingPointIndexX, *minDistanceIter, neighbors);
                            if (nextTrainingPointIndexX < trainingDatasetIndexX.cend())
                            {
                                ++nextTrainingPointIndexX;
                                distanceCalculated[1] = false;
                            }
                            else
                            {
                                distances[1] = maxDouble;
                            }
                            break;
                        case 2:
                            AddNeighbor(*prevTrainingPointIndexY, *minDistanceIter, neighbors);
                            if (prevTrainingPointIndexY > trainingDatasetIndexY.cbegin())
                            {
                                --prevTrainingPointIndexY;
                                distanceCalculated[2] = false;
                            }
                            else
                            {
                                distances[2] = maxDouble;
                            }
                            break;
                        case 3:
                            AddNeighbor(*nextTrainingPointIndexY, *minDistanceIter, neighbors);
                            if (nextTrainingPointIndexY < trainingDatasetIndexY.cend())
                            {
                                ++nextTrainingPointIndexY;
                                distanceCalculated[3] = false;
                            }
                            else
                            {
                                distances[3] = maxDouble;
                            }
                            break;
                    }
                }
            }

            auto finish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = finish - start;

            return unique_ptr<AllKnnResult>(new AllKnnResult(pNeighborsContainer, elapsed, "switchingplanesweep_serial", problem));
        }

    private:
};

#endif // SWITCHINGPLANESWEEPALGORITHM_H
