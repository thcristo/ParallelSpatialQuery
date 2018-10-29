/* This file contains a class definition of AkNN result for striped plane sweep algorithm */
#ifndef ALLKNNRESULTSTRIPES_H
#define ALLKNNRESULTSTRIPES_H

#include "AllKnnResult.h"
#include <tbb/tbb.h>
#include <cmath>

/** \brief Class definition of AkNN result for striped plane sweep algorithm
 */
class AllKnnResultStripes : public AllKnnResult
{
    public:
        AllKnnResultStripes(const AllKnnProblem& problem, const std::string& filePrefix) : AllKnnResult(problem, filePrefix)
        {
        }

        AllKnnResultStripes(const AllKnnProblem& problem, const std::string& filePrefix, bool parallelSort, bool splitByT) : AllKnnResult(problem, filePrefix),
            parallelSort(parallelSort), splitByT(splitByT)
        {
        }

        virtual ~AllKnnResultStripes() {}

        /** \brief Splits the input and training datasets into stripes
         *
         * \param numStripes size_t number of stripes to use
         * \return StripeData the structure containing data for all stripes
         *
         */
        StripeData GetStripeData(size_t numStripes)
        {
            if (!pInputDatasetStripe)
            {
                //create stripe vector for input dataset
                pInputDatasetStripe.reset(new point_vector_vector_t());
            }

            if (!pTrainingDatasetStripe)
            {
                //create stripe vector for training dataset
                pTrainingDatasetStripe.reset(new point_vector_vector_t());
            }

            if (!pStripeBoundaries)
            {
                //create vector for stripe boundaries
                pStripeBoundaries.reset(new std::vector<StripeBoundaries_t>());
            }

            //copy both datasets so we don't destroy the original problem data
            point_vector_t inputDatasetSortedY(problem.GetInputDataset());
            point_vector_t trainingDatasetSortedY(problem.GetTrainingDataset());

            if (parallelSort)
            {
                //sort by using the Intel TBB parallel sort routine
                tbb::parallel_sort(inputDatasetSortedY.begin(), inputDatasetSortedY.end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.y < point2.y;
                     });

                tbb::parallel_sort(trainingDatasetSortedY.begin(), trainingDatasetSortedY.end(),
                     [](const Point& point1, const Point& point2)
                     {
                         return point1.y < point2.y;
                     });
            }
            else
            {
                //sort by using the serial STL sort routine
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

            //check if specific number of stripes has been requested
            if (numStripes > 0)
            {
                //split datasets into this number of stripes
                create_fixed_stripes(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            }
            else
            {
                //find the optimal number of stripes
                numStripes = get_optimal_stripes();
                create_fixed_stripes(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            }

            return {*pInputDatasetStripe, *pTrainingDatasetStripe, *pStripeBoundaries};
        }


        /** \brief Returns the number of stripes
         *
         * \return size_t the number of stripes
         *
         */
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
        std::unique_ptr<point_vector_vector_t> pInputDatasetStripe;
        std::unique_ptr<point_vector_vector_t> pTrainingDatasetStripe;
        std::unique_ptr<std::vector<StripeBoundaries_t>> pStripeBoundaries;
        bool parallelSort = false;
        bool splitByT = false;

        virtual void create_fixed_stripes(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY)
        {
            //check if we want to split based on Training or input dataset
            if (splitByT)
                create_fixed_stripes_training(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            else
                create_fixed_stripes_input(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
        }

        /** \brief calculate an optimal number of stripes based on the number of training points and neighbors
         *
         * \return size_t the optimal number of stripes
         *
         */
        size_t get_optimal_stripes()
        {
            size_t numTrainingPoints = problem.GetTrainingDataset().size();
            size_t numNeighbors = problem.GetNumNeighbors();

            double numPointsPerDim = sqrt(numTrainingPoints);
            double neighborsPerDim = sqrt(numNeighbors);

            size_t optimal_stripes = llround(numPointsPerDim/neighborsPerDim);
            return optimal_stripes;
        }

        /** \brief Saves the stripes to a text file for troubleshooting reasons
         *
         * \return void
         *
         */
        void SaveStripes()
        {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss <<  "stripes_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d%H%M%S") << ".csv";

            std::ofstream outFile(ss.str(), std::ios_base::out);
            outFile.imbue(std::locale(outFile.getloc(), new punct_facet<char, ',', '.'>));

            outFile << "StripeId;MinY;MaxY;InputPoints;TrainingPoints" << std::endl;
            outFile.flush();

            size_t numStripes = pInputDatasetStripe->size();

            for (size_t i=0; i < numStripes; ++i)
            {
                outFile << i << ";" << (pStripeBoundaries->at(i)).minY  << ";" << (pStripeBoundaries->at(i)).maxY  << ";" << (pInputDatasetStripe->at(i)).size() << ";" << (pTrainingDatasetStripe->at(i)).size() << std::endl;
            }

            outFile.close();
        }

        /** \brief Splits the datasets into stripes based on the input dataset (fixed number of input points per stripe)
         *
         * \param numStripes size_t the desired number of stripes
         * \param inputDatasetSortedY const point_vector_t& the sorted input dataset
         * \param trainingDatasetSortedY const point_vector_t& the sorted training dataset
         * \return void
         *
         */
        virtual void create_fixed_stripes_input(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY)
        {
            size_t inputDatasetStripeSize = inputDatasetSortedY.size()/numStripes + 1; /**< the count of input points  per stripe */
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();

            auto inputIterStart = inputDatasetSortedY.cbegin();
            auto inputIterEnd = inputIterStart + inputDatasetStripeSize;
            auto trainingIterStart = trainingDatasetSortedY.cbegin();

            bool exit = false;

            //loop through input dataset, each step should contain the same number of input points
            do
            {
                //while there are any input points having same y, advance the end of the stripe to the next input point
                while (inputIterEnd < inputDatasetSortedYEnd && prev(inputIterEnd)->y == inputIterEnd->y)
                {
                    ++inputIterEnd;
                }

                //we found input points for the current stripe
                pInputDatasetStripe->push_back(point_vector_t(inputIterStart, inputIterEnd));

                //miny boundary for current stripe
                double minY = inputIterStart->y <= trainingIterStart->y ? inputIterStart->y : trainingIterStart->y;

                //sort input points of current stripe by x
                if (parallelSort)
                {
                    tbb::parallel_sort(pInputDatasetStripe->back().begin(), pInputDatasetStripe->back().end(),
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

                //now find the maxy boundary of current stripe
                double maxY = minY;

                //find the training point for current stripe
                if (trainingIterStart < trainingDatasetSortedYEnd)
                {
                    auto trainingIterEnd = trainingDatasetSortedYEnd;
                    if (inputIterEnd != inputDatasetSortedYEnd)
                    {
                        trainingIterEnd = trainingIterStart;
                        double ylimit = prev(inputIterEnd)->y;
                        //while there are points with same y, advance the end of the current stripe
                        while (trainingIterEnd < trainingDatasetSortedYEnd && trainingIterEnd->y <= ylimit)
                            ++trainingIterEnd;
                    }

                    //we found training points for current stripe
                    pTrainingDatasetStripe->push_back(point_vector_t(trainingIterStart, trainingIterEnd));

                    maxY = prev(trainingIterEnd)->y >= prev(inputIterEnd)->y ? prev(trainingIterEnd)->y : prev(inputIterEnd)->y;

                    //sort training points of current stripe by x
                    if (parallelSort)
                    {
                        tbb::parallel_sort(pTrainingDatasetStripe->back().begin(), pTrainingDatasetStripe->back().end(),
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

                    //start of next stripe is the end of current stripe
                    trainingIterStart = trainingIterEnd;
                }
                else
                {
                    //the current stripe does not contain any training points
                    pTrainingDatasetStripe->push_back(point_vector_t());

                    maxY = prev(inputIterEnd)->y;
                }

                //keep the boundaries of current stripe in a vector
                pStripeBoundaries->push_back({minY, maxY});

                //calculate the starting point for next stripe
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
                    //we have examined all input points
                    exit = true;
                }

            } while (!exit);
        }

        /** \brief Splits the datasets into stripes based on the training dataset (fixed number of training points per stripe)
         *
         * \param numStripes size_t the desired number of stripes
         * \param inputDatasetSortedY const point_vector_t& the sorted input dataset
         * \param trainingDatasetSortedY const point_vector_t& the sorted training dataset
         * \return void
         *
         */
        virtual void create_fixed_stripes_training(size_t numStripes, const point_vector_t& inputDatasetSortedY, const point_vector_t& trainingDatasetSortedY)
        {
            //The implementation is exactly the same as create_fixed_stripes_input
            //with the only difference of switching between input and training datasets

            size_t trainingDatasetStripeSize = trainingDatasetSortedY.size()/numStripes + 1;
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();

            auto trainingIterStart = trainingDatasetSortedY.cbegin();
            auto trainingIterEnd = trainingIterStart + trainingDatasetStripeSize;
            auto inputIterStart = inputDatasetSortedY.cbegin();

            bool exit = false;

            do
            {
                while (trainingIterEnd < trainingDatasetSortedYEnd && prev(trainingIterEnd)->y == trainingIterEnd->y)
                {
                    ++trainingIterEnd;
                }

                pTrainingDatasetStripe->push_back(point_vector_t(trainingIterStart, trainingIterEnd));

                double minY = inputIterStart->y <= trainingIterStart->y ? inputIterStart->y : trainingIterStart->y;

                if (parallelSort)
                {
                    tbb::parallel_sort(pTrainingDatasetStripe->back().begin(), pTrainingDatasetStripe->back().end(),
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

                double maxY = minY;

                if (inputIterStart < inputDatasetSortedYEnd)
                {
                    auto inputIterEnd = inputDatasetSortedYEnd;
                    if (trainingIterEnd != trainingDatasetSortedYEnd)
                    {
                        inputIterEnd = inputIterStart;
                        double ylimit = prev(trainingIterEnd)->y;
                        while (inputIterEnd < inputDatasetSortedYEnd && inputIterEnd->y <= ylimit)
                            ++inputIterEnd;
                    }

                    pInputDatasetStripe->push_back(point_vector_t(inputIterStart, inputIterEnd));

                    maxY = prev(inputIterEnd)->y >= prev(trainingIterEnd)->y ? prev(inputIterEnd)->y : prev(trainingIterEnd)->y;

                    if (parallelSort)
                    {
                        tbb::parallel_sort(pInputDatasetStripe->back().begin(), pInputDatasetStripe->back().end(),
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

                    inputIterStart = inputIterEnd;
                }
                else
                {
                    pInputDatasetStripe->push_back(point_vector_t());

                    maxY = prev(trainingIterEnd)->y;
                }

                pStripeBoundaries->push_back({minY, maxY});

                if (trainingIterEnd < trainingDatasetSortedYEnd)
                {
                    trainingIterStart = trainingIterEnd;
                    if ((size_t)distance(trainingIterStart, trainingDatasetSortedYEnd) >= trainingDatasetStripeSize)
                    {
                        trainingIterEnd = trainingIterStart + trainingDatasetStripeSize;
                    }
                    else
                    {
                        trainingIterEnd = trainingDatasetSortedYEnd;
                    }
                }
                else
                {
                    exit = true;
                }

            } while (!exit);
        }
};

#endif // ALLKNNRESULTSTRIPES_H
