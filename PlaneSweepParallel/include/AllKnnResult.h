#ifndef AllKnnRESULT_H
#define AllKnnRESULT_H

#include <memory>
#include <chrono>
#include <fstream>
#include <cmath>
#include "PlaneSweepParallel.h"
#include "AllKnnProblem.h"
#include "PointNeighbors.h"

class AllKnnResult
{
    public:
        AllKnnResult(const AllKnnProblem& problem, const string& filePrefix)
                     : problem(problem), filePrefix(filePrefix)
        {
        }

        AllKnnResult(const AllKnnProblem& problem, const string& filePrefix,
                     unique_ptr<pointNeighbors_priority_queue_vector_t>& pNeighborsContainer,
                     const chrono::duration<double>& elapsed, const chrono::duration<double>& elapsedSorting)
                     :  problem(problem), filePrefix(filePrefix),
                        pNeighborsPriorityQueueVector(move(pNeighborsContainer)), elapsed(elapsed), elapsedSorting(elapsedSorting)
        {
            CalcHeapStats();
        }

        virtual ~AllKnnResult() {}

        const chrono::duration<double>& getDuration() const { return elapsed; }

        void setDuration(chrono::duration<double> value)
        {
            elapsed = value;
        }

        const chrono::duration<double>& getDurationSorting() const { return elapsedSorting; }

        virtual const chrono::duration<double> getDurationCommitWindow() const
        {
            return chrono::duration<double>(0.0);
        }

        virtual const chrono::duration<double> getDurationFinalSorting() const
        {
            return chrono::duration<double>(0.0);
        }

        virtual size_t getNumFirstPassWindows() const
        {
            return 0;
        }

        virtual size_t getNumSecondPassWindows() const
        {
            return 0;
        }

        void setDurationSorting(chrono::duration<double> value)
        {
            elapsedSorting = value;
        }

        void setNeighborsContainer(unique_ptr<pointNeighbors_priority_queue_vector_t>& pNeighborsContainer)
        {
            pNeighborsPriorityQueueVector = move(pNeighborsContainer);
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

        virtual void SaveToFile() const
        {
            auto ms = chrono::duration_cast<chrono::milliseconds>(elapsed);

            auto now = chrono::system_clock::now();
            auto in_time_t = chrono::system_clock::to_time_t(now);

            stringstream ss;
            ss << filePrefix << "_" << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << "_" << ms.count() << ".txt";

            ofstream outFile(ss.str(), ios_base::out);

            const point_vector_t& inputDataset = problem.GetInputDataset();

            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                outFile << inputPoint->id;

                NeighborsEnumerator* pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);

                vector<Neighbor> removedNeighbors;

                while (pNeighbors->HasNext())
                {
                    Neighbor neighbor = pNeighbors->Next();
                    removedNeighbors.push_back(neighbor);

                    if (neighbor.pointId > 0)
                    {
                        outFile << "\t(" << neighbor.pointId << " " << neighbor.distanceSquared << ")";
                    }
                    else
                    {
                        outFile << "\t(" << "NULL" << " " << neighbor.distanceSquared << ")";
                    }
                }

                outFile << endl;

                pNeighbors->AddAllRemoved(removedNeighbors);
            }

            outFile.close();
        }

        virtual unique_ptr<vector<unsigned long>> FindDifferences(AllKnnResult& result, double accuracy)
        {
            auto differences = unique_ptr<vector<unsigned long>>(new vector<unsigned long>());

            auto& inputDataset = problem.GetInputDataset();

            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                NeighborsEnumerator* pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);
                NeighborsEnumerator* pNeighborsReference = &result.pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);

                vector<Neighbor> removedNeighbors;
                vector<Neighbor> removedNeighborsReference;

                while (pNeighbors->HasNext())
                {
                    Neighbor neighbor = pNeighbors->Next();
                    removedNeighbors.push_back(neighbor);

                    if (pNeighborsReference->HasNext())
                    {
                        Neighbor neighborReference = pNeighborsReference->Next();
                        removedNeighborsReference.push_back(neighborReference);

                        double diff = neighbor.distanceSquared - neighborReference.distanceSquared;

                        if (abs(diff) > accuracy)
                        {
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

                pNeighbors->AddAllRemoved(removedNeighbors);
                pNeighborsReference->AddAllRemoved(removedNeighborsReference);
            }

            return differences;
        }

        pointNeighbors_priority_queue_vector_t& GetNeighborsPriorityQueueVector()
        {
            return *pNeighborsPriorityQueueVector;
        }

        virtual void CalcHeapStats()
        {
            minHeapAdditions = numeric_limits<size_t>::max();
            maxHeapAdditions = 0;
            totalHeapAdditions = 0;

            size_t numInputPoints = pNeighborsPriorityQueueVector->size();

            for (size_t i=0; i < numInputPoints;  ++i)
            {
                auto& neighbors = pNeighborsPriorityQueueVector->at(i);
                auto additions = neighbors.GetNumAdditions();
                if (additions < minHeapAdditions)
                {
                        minHeapAdditions = additions;
                }
                if (additions > maxHeapAdditions)
                {
                        maxHeapAdditions = additions;
                }
                totalHeapAdditions += additions;
            }

            avgHeapAdditions = (1.0*totalHeapAdditions)/numInputPoints;
        }

    protected:
        const AllKnnProblem& problem;
        string filePrefix;
        size_t minHeapAdditions = 0;
        size_t maxHeapAdditions = 0;
        double avgHeapAdditions = 0.0;
        size_t totalHeapAdditions = 0.0;

    private:
        unique_ptr<pointNeighbors_priority_queue_vector_t> pNeighborsPriorityQueueVector;
        chrono::duration<double> elapsed;
        chrono::duration<double> elapsedSorting;

};

#endif // AllKnnRESULT_H
