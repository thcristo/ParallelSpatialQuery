#ifndef AllKnnRESULT_H
#define AllKnnRESULT_H

#include <memory>
#include <chrono>
#include <fstream>
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

        void setNeighborsContainer(unique_ptr<neighbors_priority_queue_container_t>& pNeighborsContainer)
        {
            pNeighborsPriorityQueueContainer = move(pNeighborsContainer);
        }

        void setNeighborsContainer(unique_ptr<pointNeighbors_priority_queue_vector_t>& pNeighborsContainer)
        {
            pNeighborsPriorityQueueVector = move(pNeighborsContainer);
        }

        void SaveToFile() const
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

                NeighborsEnumerator* pNeighbors = nullptr;
                if (pNeighborsPriorityQueueVector != nullptr)
                {
                    pNeighbors = &pNeighborsPriorityQueueVector->at((inputPoint->id) - 1);
                }
                else if (pNeighborsPriorityQueueContainer != nullptr)
                {
                    pNeighbors = &pNeighborsPriorityQueueContainer->at(inputPoint->id);
                }

                while (pNeighbors->HasNext())
                {
                    Neighbor neighbor = pNeighbors->Next();
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
            }

            outFile.close();
        }
    protected:
        const AllKnnProblem& problem;

    private:
        string filePrefix;
        unique_ptr<neighbors_priority_queue_container_t> pNeighborsPriorityQueueContainer;
        unique_ptr<pointNeighbors_priority_queue_vector_t> pNeighborsPriorityQueueVector;
        chrono::duration<double> elapsed;
        chrono::duration<double> elapsedSorting;
};

#endif // AllKnnRESULT_H
