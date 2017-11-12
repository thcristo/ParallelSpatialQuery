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
        AllKnnResult(unique_ptr<neighbors_priority_queue_container_t>& pNeighborsContainer,
                     const chrono::duration<double>& elapsed, const chrono::duration<double>& elapsedSorting,
                     const string& filePrefix, const AllKnnProblem& problem)
                     : pNeighborsPriorityQueueContainer(move(pNeighborsContainer)), elapsed(elapsed), elapsedSorting(elapsedSorting),
                        filePrefix(filePrefix), problem(problem)
        {

        }

        AllKnnResult(unique_ptr<neighbors_vector_container_t>& pNeighborsContainer,
                     const chrono::duration<double>& elapsed, const chrono::duration<double>& elapsedSorting,
                     const string& filePrefix, const AllKnnProblem& problem)
                     : pNeighborsVectorContainer(move(pNeighborsContainer)), elapsed(elapsed), elapsedSorting(elapsedSorting),
                        filePrefix(filePrefix), problem(problem)
        {

        }

        virtual ~AllKnnResult() {}

        const chrono::duration<double>& duration() const { return elapsed; }
        const chrono::duration<double>& durationSorting() const { return elapsedSorting; }

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
                if (pNeighborsPriorityQueueContainer != nullptr)
                {
                    pNeighbors = &pNeighborsPriorityQueueContainer->at(inputPoint->id);
                }
                else if (pNeighborsVectorContainer != nullptr)
                {
                    pNeighbors = &pNeighborsVectorContainer->at(inputPoint->id);
                }

                while (pNeighbors->HasNext())
                {
                   Neighbor neighbor = pNeighbors->Next();
                   outFile << "\t(" << neighbor.point->id << " " << neighbor.distanceSquared << ")";
                }

                outFile << endl;
            }

            outFile.close();
        }
    protected:

    private:
        unique_ptr<neighbors_priority_queue_container_t> pNeighborsPriorityQueueContainer;
        unique_ptr<neighbors_vector_container_t> pNeighborsVectorContainer;
        chrono::duration<double> elapsed;
        chrono::duration<double> elapsedSorting;
        string filePrefix;
        const AllKnnProblem& problem;
};

#endif // AllKnnRESULT_H
