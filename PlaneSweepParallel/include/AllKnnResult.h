/* This file contains the class definition of the AkNN algorithm result */

#ifndef AllKnnRESULT_H
#define AllKnnRESULT_H

#include <memory>
#include <chrono>
#include <fstream>
#include <cmath>
#include "PlaneSweepParallel.h"
#include "AllKnnProblem.h"
#include "PointNeighbors.h"

/** \brief Class definition of AkNN result
 */
class AllKnnResult
{
    public:
        AllKnnResult(const AllKnnProblem& problem, const std::string& filePrefix)
                     : problem(problem), filePrefix(filePrefix)
        {
        }

        AllKnnResult(const AllKnnProblem& problem, const std::string& filePrefix,
                     std::unique_ptr<pointNeighbors_priority_queue_vector_t>& pNeighborsContainer,
                     const std::chrono::duration<double>& elapsed, const std::chrono::duration<double>& elapsedSorting)
                     :  problem(problem), filePrefix(filePrefix),
                        pNeighborsPriorityQueueVector(std::move(pNeighborsContainer)), elapsed(elapsed), elapsedSorting(elapsedSorting)
        {
            //calculate heap statistics (additions etc.) for reporting purposes
            CalcHeapStats();
        }

        virtual ~AllKnnResult() {}

        const std::chrono::duration<double>& getDuration() const { return elapsed; }

        void setDuration(std::chrono::duration<double> value)
        {
            elapsed = value;
        }

        const std::chrono::duration<double>& getDurationSorting() const { return elapsedSorting; }

        virtual const std::chrono::duration<double> getDurationCommitWindow() const
        {
            return std::chrono::duration<double>(0.0);
        }

        virtual const std::chrono::duration<double> getDurationFinalSorting() const
        {
            return std::chrono::duration<double>(0.0);
        }

        virtual size_t getNumFirstPassWindows() const
        {
            return 0;
        }

        virtual size_t getNumSecondPassWindows() const
        {
            return 0;
        }

        void setDurationSorting(std::chrono::duration<double> value)
        {
            elapsedSorting = value;
        }

        void setNeighborsContainer(std::unique_ptr<pointNeighbors_priority_queue_vector_t>& pNeighborsContainer)
        {
            pNeighborsPriorityQueueVector = std::move(pNeighborsContainer);
            CalcHeapStats();
        }

        size_t getMinHeapAdditions()
        {
            return minHeapAdditions;
        }

        size_t getMaxHeapAdditions()
        {
            return maxHeapAdditions;
        }

        double getAvgHeapAdditions()
        {
            return avgHeapAdditions;
        }

        size_t getTotalHeapAdditions()
        {
            return totalHeapAdditions;
        }

        virtual size_t getNumStripes()
        {
           return 0;
        }

        virtual size_t getNumPendingPoints()
        {
            return 0;
        }

        virtual bool HasAllocationError()
        {
            return false;
        }

        /** \brief Saves neighbors found for each input point to a text file
         *
         */
        virtual void SaveToFile() const
        {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);

            //generate a unique filename
            std::stringstream ss;
            ss << filePrefix << "_" << std::put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << "_" << ms.count() << ".txt";

            std::ofstream outFile(ss.str(), std::ios_base::out);

            const point_vector_t& inputDataset = problem.GetInputDataset();

            //loop through the input dataset
            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                outFile << inputPoint->id;

                NeighborsEnumerator* pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);

                std::vector<Neighbor> removedNeighbors;

                //loop through all the neighbors in the max heap
                while (pNeighbors->HasNext())
                {
                    Neighbor neighbor = pNeighbors->Next();
                    //add neighbor to a second vector so we can put it back again
                    //(this is needed when we want to use the result as a reference for comparison)
                    removedNeighbors.push_back(neighbor);

                    //output point Id and squared distance
                    if (neighbor.pointId > 0)
                    {
                        outFile << "\t(" << neighbor.pointId << " " << neighbor.distanceSquared << ")";
                    }
                    else
                    {
                        outFile << "\t(" << "NULL" << " " << neighbor.distanceSquared << ")";
                    }
                }

                outFile << std::endl;

                //put back all removed neighbors
                pNeighbors->AddAllRemoved(removedNeighbors);
            }

            outFile.close();
        }

        /** \brief Compares a result with a reference result to find any differences in distances of neighbors
         *
         * \param result AllKnnResult& the result to check for differences
         * \param accuracy double the accuracy to use for comparisons
         * \return unique_ptr<vector<unsigned long>>  vector of input point ids where differences exist
         *
         */
        virtual std::unique_ptr<std::vector<unsigned long>> FindDifferences(AllKnnResult& result, double accuracy)
        {
            auto differences = std::unique_ptr<std::vector<unsigned long>>(new std::vector<unsigned long>());

            auto& inputDataset = problem.GetInputDataset();

            //loop through all input points
            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                NeighborsEnumerator* pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);
                NeighborsEnumerator* pNeighborsReference = &result.pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);

                std::vector<Neighbor> removedNeighbors;
                std::vector<Neighbor> removedNeighborsReference;

                //loop through all neighbors
                while (pNeighbors->HasNext())
                {
                    Neighbor neighbor = pNeighbors->Next();
                    removedNeighbors.push_back(neighbor);

                    if (pNeighborsReference->HasNext())
                    {
                        Neighbor neighborReference = pNeighborsReference->Next();
                        removedNeighborsReference.push_back(neighborReference);

                        //compare with reference result and check if difference in squared distance exceeds the desired accuracy
                        double diff = neighbor.distanceSquared - neighborReference.distanceSquared;

                        if (abs(diff) > accuracy)
                        {
                            //insert the id of input point in a vector for reporting purposes
                            differences->push_back(inputPoint->id);
                            break;
                        }
                    }
                    else
                    {
                        differences->push_back(inputPoint->id);
                        break;
                    }
                }

                if (!pNeighbors->HasNext() && pNeighborsReference->HasNext())
                {
                    differences->push_back(inputPoint->id);
                }

                //put back all removed neighbors from both vectors
                pNeighbors->AddAllRemoved(removedNeighbors);
                pNeighborsReference->AddAllRemoved(removedNeighborsReference);
            }

            return differences;
        }

        pointNeighbors_priority_queue_vector_t& GetNeighborsPriorityQueueVector()
        {
            return *pNeighborsPriorityQueueVector;
        }

        /** \brief Calculates heap statistics for reporting purposes
         */
        virtual void CalcHeapStats()
        {
            minHeapAdditions = std::numeric_limits<size_t>::max();
            maxHeapAdditions = 0;
            totalHeapAdditions = 0;

            size_t numInputPoints = pNeighborsPriorityQueueVector->size();

            //loop through all input points
            for (size_t i=0; i < numInputPoints;  ++i)
            {
                auto& neighbors = pNeighborsPriorityQueueVector->at(i);
                auto additions = neighbors.GetNumAdditions();

                //find the minimum additions to max heap
                if (additions < minHeapAdditions)
                {
                        minHeapAdditions = additions;
                }

                //find the maximum additions to max heap
                if (additions > maxHeapAdditions)
                {
                        maxHeapAdditions = additions;
                }

                //find the total additions to max heap
                totalHeapAdditions += additions;
            }

            //find the average additions to max heap
            avgHeapAdditions = (1.0*totalHeapAdditions)/numInputPoints;
        }

    protected:
        const AllKnnProblem& problem;
        std::string filePrefix;
        size_t minHeapAdditions = 0;
        size_t maxHeapAdditions = 0;
        double avgHeapAdditions = 0.0;
        size_t totalHeapAdditions = 0.0;

    private:
        std::unique_ptr<pointNeighbors_priority_queue_vector_t> pNeighborsPriorityQueueVector;
        std::chrono::duration<double> elapsed;
        std::chrono::duration<double> elapsedSorting;

};

#endif // AllKnnRESULT_H
