#ifndef AllKnnRESULT_H
#define AllKnnRESULT_H

#include <memory>
#include "PlaneSweepParallel.h"
#include <chrono>
#include <fstream>

using namespace std;

class AllKnnResult
{
    public:
        AllKnnResult(unique_ptr<neighbors_container_t>& ppNeighborsContainer, const chrono::duration<double>& elapsed,
                     const string& filePrefix) : elapsed(elapsed), filePrefix(filePrefix)
        {
            this->pNeighborsContainer = move(pNeighborsContainer);
        }

        virtual ~AllKnnResult() {}

        const unique_ptr<neighbors_container_t>& GetResultContainer() const
        {
            return pNeighborsContainer;
        }

        const chrono::duration<double>& duration() const { return elapsed; }

        void SaveToFile() const
        {
            auto ms = chrono::duration_cast<chrono::milliseconds>(elapsed);

            auto now = chrono::system_clock::now();
            auto in_time_t = chrono::system_clock::to_time_t(now);

            stringstream ss;
            ss << filePrefix << "_" << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << "_" << ms.count() << ".txt";

            ofstream outFile(ss.str(), ios_base::out);

            for (auto element = pNeighborsContainer->cbegin(); element != pNeighborsContainer->cend(); ++element)
            {
               outFile << element->first;

               auto& neighbors = element->second;

               for (auto neighbor = neighbors.cbegin(); neighbor != neighbors.cend(); ++neighbor)
               {
                   outFile << "\t(" << neighbor->point->id << "," << neighbor->distanceSquared << ")";
               }

               outFile << endl;
            }

            outFile.close();
        }
    protected:

    private:
        unique_ptr<neighbors_container_t> pNeighborsContainer;
        chrono::duration<double> elapsed;
        string filePrefix;
};

#endif // AllKnnRESULT_H
