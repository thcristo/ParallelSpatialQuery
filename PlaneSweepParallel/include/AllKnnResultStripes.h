#ifndef ALLKNNRESULTSTRIPES_H
#define ALLKNNRESULTSTRIPES_H

#include "AllKnnResult.h"
#include <tbb/tbb.h>
#include <cmath>

using namespace tbb;


class AllKnnResultStripes : public AllKnnResult
{
    public:
        AllKnnResultStripes(const AllKnnProblem& problem, const string& filePrefix) : AllKnnResult(problem, filePrefix)
        {
        }

        AllKnnResultStripes(const AllKnnProblem& problem, const string& filePrefix, bool parallelSort) : AllKnnResult(problem, filePrefix),
            parallelSort(parallelSort)
        {
        }

        virtual ~AllKnnResultStripes() {}

        StripeData GetStripeData(size_t numStripes)
        {
            if (!pInputDatasetStripe)
            {
                pInputDatasetStripe.reset(new point_vector_vector_t());
            }

            if (!pTrainingDatasetStripe)
            {
                pTrainingDatasetStripe.reset(new point_vector_vector_t());
            }

            if (!pStripeBoundaries)
            {
                pStripeBoundaries.reset(new vector<StripeBoundaries_t>());
            }

            point_vector_t inputDatasetSortedY(problem.GetInputDataset());
            point_vector_t trainingDatasetSortedY(problem.GetTrainingDataset());

            if (parallelSort)
            {
                parallel_sort(inputDatasetSortedY.begin(), inputDatasetSortedY.end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.y < point2.y;
                     });

                parallel_sort(trainingDatasetSortedY.begin(), trainingDatasetSortedY.end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.y < point2.y;
                     });
            }
            else
            {
                sort(inputDatasetSortedY.begin(), inputDatasetSortedY.end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.y < point2.y;
                     });

                sort(trainingDatasetSortedY.begin(), trainingDatasetSortedY.end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.y < point2.y;
                     });
            }

            if (numStripes > 0)
            {
                create_fixed_stripes(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            }
            else
            {
                numStripes = get_optimal_stripes();
                create_fixed_stripes(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            }

            return {*pInputDatasetStripe, *pTrainingDatasetStripe, *pStripeBoundaries};
        }

        size_t getNumStripes() override
        {
            if (pInputDatasetStripe != nullptr)
            {
                return pInputDatasetStripe->size();
            }
            else
            {
                return 0;
            }
        }
    protected:

        virtual void create_fixed_stripes(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY)
        {
            size_t inputDatasetStripeSize = inputDatasetSortedY.size()/numStripes + 1;
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();

            auto inputIterStart = inputDatasetSortedY.cbegin();
            auto inputIterEnd = inputIterStart + inputDatasetStripeSize;
            auto trainingIterStart = trainingDatasetSortedY.cbegin();

            bool exit = false;

            do
            {
                while (inputIterEnd < inputDatasetSortedYEnd && prev(inputIterEnd)->y == inputIterEnd->y)
                {
                    ++inputIterEnd;
                }

                pInputDatasetStripe->push_back(point_vector_t(inputIterStart, inputIterEnd));

                double minY = inputIterStart->y <= trainingIterStart->y ? inputIterStart->y : trainingIterStart->y;

                if (parallelSort)
                {
                    parallel_sort(pInputDatasetStripe->back().begin(), pInputDatasetStripe->back().end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }
                else
                {
                    sort(pInputDatasetStripe->back().begin(), pInputDatasetStripe->back().end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                }

                double maxY = minY;

                if (trainingIterStart < trainingDatasetSortedYEnd)
                {
                    auto trainingIterEnd = inputIterEnd == inputDatasetSortedYEnd ? trainingDatasetSortedYEnd :
                                            upper_bound(trainingIterStart, trainingDatasetSortedYEnd, prev(inputIterEnd)->y,
                                                          [](const double& value, const Point& point) { return value < point.y; } );

                    pTrainingDatasetStripe->push_back(point_vector_t(trainingIterStart, trainingIterEnd));

                    maxY = prev(trainingIterEnd)->y >= prev(inputIterEnd)->y ? prev(trainingIterEnd)->y : prev(inputIterEnd)->y;

                    if (parallelSort)
                    {
                        parallel_sort(pTrainingDatasetStripe->back().begin(), pTrainingDatasetStripe->back().end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                    }
                    else
                    {
                        sort(pTrainingDatasetStripe->back().begin(), pTrainingDatasetStripe->back().end(),
                         [](const Point& point1, const Point& point2)
                         {
                             return point1.x < point2.x;
                         });
                    }

                    trainingIterStart = trainingIterEnd;
                }
                else
                {
                    pTrainingDatasetStripe->push_back(point_vector_t());

                    maxY = prev(inputIterEnd)->y;
                }

                pStripeBoundaries->push_back({minY, maxY});

                if (inputIterEnd < inputDatasetSortedYEnd)
                {
                    inputIterStart = inputIterEnd;
                    if ((size_t)distance(inputIterStart, inputDatasetSortedYEnd) >= inputDatasetStripeSize)
                    {
                        inputIterEnd = inputIterStart + inputDatasetStripeSize;
                    }
                    else
                    {
                        inputIterEnd = inputDatasetSortedYEnd;
                    }
                }
                else
                {
                    exit = true;
                }

            } while (!exit);
        }

        unique_ptr<point_vector_vector_t> pInputDatasetStripe;
        unique_ptr<point_vector_vector_t> pTrainingDatasetStripe;
        unique_ptr<vector<StripeBoundaries_t>> pStripeBoundaries;

        bool parallelSort = false;

        size_t get_optimal_stripes()
        {
            size_t numTrainingPoints = problem.GetTrainingDataset().size();
            size_t numNeighbors = problem.GetNumNeighbors();

            double numPointsPerDim = sqrt(numTrainingPoints);
            double neighborsPerDim = sqrt(numNeighbors);

            size_t optimal_stripes = llround(numPointsPerDim/neighborsPerDim);
            return optimal_stripes;
        }

        void SaveStripes()
        {
            auto now = chrono::system_clock::now();
            auto in_time_t = chrono::system_clock::to_time_t(now);
            stringstream ss;
            ss <<  "stripes_" << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << ".csv";

            ofstream outFile(ss.str(), ios_base::out);
            outFile.imbue(locale(outFile.getloc(), new punct_facet<char, ',', '.'>));

            outFile << "StripeId;MinY;MaxY;InputPoints;TrainingPoints" << endl;
            outFile.flush();

            size_t numStripes = pInputDatasetStripe->size();

            for (size_t i=0; i < numStripes; ++i)
            {
                outFile << i << ";" << (pStripeBoundaries->at(i)).minY  << ";" << (pStripeBoundaries->at(i)).maxY  << ";" << (pInputDatasetStripe->at(i)).size() << ";" << (pTrainingDatasetStripe->at(i)).size() << endl;
            }

            outFile.close();
        }
};

#endif // ALLKNNRESULTSTRIPES_H
