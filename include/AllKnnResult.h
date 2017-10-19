#ifndef AllKnnRESULT_H
#define AllKnnRESULT_H

#include <memory>
#include "PlaneSweepParallel.h"

using namespace std;

class AllKnnResult
{
    public:
        AllKnnResult(unique_ptr<neighbors_container_t>& ppNeighborsContainer)
        {
            this->pNeighborsContainer = move(pNeighborsContainer);
        }
        virtual ~AllKnnResult() {}

        const unique_ptr<neighbors_container_t>& GetResultContainer() const
        {
            return pNeighborsContainer;
        }
    protected:

    private:
        unique_ptr<neighbors_container_t> pNeighborsContainer;
};

#endif // AllKnnRESULT_H
