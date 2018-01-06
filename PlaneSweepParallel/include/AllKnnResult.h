#ifndef AllKnnRESULT_H
#define AllKnnRESULT_H

#include <memory>
#include <chrono>
#include <fstream>
#include <cmath>
//#include "PlaneSweepParallel.h"
#include "PointNeighbors.h"

template<class ProblemT, class NeighborsContainerT, class PointVectorT, class PointIdVectorT>
class AllKnnResult
{
    public:
        AllKnnResult(const ProblemT& problem, const string& filePrefix)
                     : problem(problem), filePrefix(filePrefix)
        {
        }

        AllKnnResult(const ProblemT& problem, const string& filePrefix,
                     unique_ptr<NeighborsContainerT>& pNeighborsContainer,
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

        void setDurationSorting(chrono::duration<double> value)
        {
            elapsedSorting = value;
        }

        /*
        void setNeighborsContainer(unique_ptr<neighbors_priority_queue_container_t>& pNeighborsContainer)
        {
            pNeighborsPriorityQueueContainer = move(pNeighborsContainer);
        }
        */
        void setNeighborsContainer(unique_ptr<NeighborsContainerT>& pNeighborsContainer)
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

        void SaveToFile() const
        {
            auto ms = chrono::duration_cast<chrono::milliseconds>(elapsed);

            auto now = chrono::system_clock::now();
            auto in_time_t = chrono::system_clock::to_time_t(now);

            stringstream ss;
            ss << filePrefix << "_" << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << "_" << ms.count() << ".txt";

            ofstream outFile(ss.str(), ios_base::out);

            const PointVectorT& inputDataset = problem.GetInputDataset();

            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                outFile << inputPoint->id;

                /*
                NeighborsEnumerator* pNeighbors = nullptr;
                if (pNeighborsPriorityQueueVector != nullptr)
                {
                    pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);
                }
                else if (pNeighborsPriorityQueueContainer != nullptr)
                {
                    pNeighbors = &pNeighborsPriorityQueueContainer->at(inputPoint->id);
                }
                */

                NeighborsEnumerator* pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);

                vector<Neighbor> removedNeighbors;

                while (pNeighbors->HasNext())
                {
                    Neighbor neighbor = pNeighbors->Next();
                    removedNeighbors.push_back(neighbor);

                    if (neighbor.point != nullptr)
                    {
                        outFile << "\t(" << neighbor.point->id << " " << neighbor.distanceSquared << ")";
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

        unique_ptr<PointIdVectorT> FindDifferences(const AllKnnResult<ProblemT, NeighborsContainerT, PointVectorT, PointIdVectorT>& result,
                                                   double accuracy)
        {
            auto differences = unique_ptr<PointIdVectorT>(new PointIdVectorT());

            auto& inputDataset = problem.GetInputDataset();

            for (auto inputPoint = inputDataset.cbegin(); inputPoint != inputDataset.cend(); ++inputPoint)
            {
                /*
                NeighborsEnumerator* pNeighbors = nullptr;
                if (pNeighborsPriorityQueueVector != nullptr)
                {
                    pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);
                }
                else if (pNeighborsPriorityQueueContainer != nullptr)
                {
                    pNeighbors = &pNeighborsPriorityQueueContainer->at(inputPoint->id);
                }
                */

                NeighborsEnumerator* pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);

                /*
                NeighborsEnumerator* pNeighborsReference = nullptr;
                if (result.pNeighborsPriorityQueueVector != nullptr)
                {
                    pNeighborsReference = &result.pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);
                }
                else if (result.pNeighborsPriorityQueueContainer != nullptr)
                {
                    pNeighborsReference = &result.pNeighborsPriorityQueueContainer->at(inputPoint->id);
                }
                */
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
    protected:
        const ProblemT& problem;

    private:
        string filePrefix;
        //unique_ptr<neighbors_priority_queue_container_t> pNeighborsPriorityQueueContainer;
        unique_ptr<NeighborsContainerT> pNeighborsPriorityQueueVector;
        chrono::duration<double> elapsed;
        chrono::duration<double> elapsedSorting;
        size_t minHeapAdditions = 0;
        size_t maxHeapAdditions = 0;
        double avgHeapAdditions = 0.0;
        size_t totalHeapAdditions = 0.0;

        void CalcHeapStats()
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
};

#endif // AllKnnRESULT_H
