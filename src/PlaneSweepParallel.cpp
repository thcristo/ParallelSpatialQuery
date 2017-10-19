#include <iostream>
#include <cstdlib>
#include <exception>
#include "ParallelPlaneSweepAlgorithm.h"
#include "BruteForceAlgorithm.h"
#include "AllKnnProblem.h"
#include "AllKnnResult.h"



using namespace std;


int main(int argc, char* argv[])
{

    if (argc != 4)
    {
        cout << "Argument error. Please enter:\n";
        cout << "Argument 1: The number of the nearest neighbors of the query,\n";
        cout << "Argument 2: The file of the input dataset,\n";
        cout << "Argument 3: The file of the training dataset,\n";
        return 1;
    }

    int numNeighbors = atoi(argv[1]);

    try
    {
        AllKnnProblem problem(argv[2], argv[3], numNeighbors);

        cout << "Read " << problem.GetInputDataset().size() << " input points and " << problem.GetTrainingDataset().size() << " training points." << endl;

        BruteForceAlgorithm bruteForce;
        unique_ptr<AllKnnResult> pResult = bruteForce.Process(problem);

        return 0;
    }
    catch(exception& ex)
    {
        cout << "Exception: " << ex.what() << endl;
        return 1;
    }
}
