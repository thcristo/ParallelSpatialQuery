#include <iostream>

#include "ParallelPlaneSweepAlgorithm.h"
#include "AllKnnProblem.h"
#include "AllKnnResult.h"



using namespace std;


int main(int argc, char *argv[]){

    if (argc != 4)
    {
        cout << "Argument error. Please enter:\n";
        cout << "Argument 1: The number of the nearest neighbors of the query,\n";
        cout << "Argument 2: The file of the input dataset,\n";
        cout << "Argument 3: The file of the training dataset,\n";
        return 1;
    }

    int numNeighbors = atoi(argv[1]);
    return 0;
}
