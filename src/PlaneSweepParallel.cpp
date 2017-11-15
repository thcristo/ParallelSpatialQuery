#include <iostream>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <chrono>
#include "ParallelPlaneSweepAlgorithm.h"
#include "BruteForceAlgorithm.h"
#include "BruteForceParallelAlgorithm.h"
#include "BruteForceParallelTBBAlgorithm.h"
#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "PlaneSweepAlgorithm.h"
#include "SwitchingPlaneSweepAlgorithm.h"

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
        unique_ptr<AllKnnResult> pResult;

/*
        BruteForceAlgorithm bruteForce;
        pResult = bruteForce.Process(problem);
        cout << fixed << setprecision(3) << "Brute force duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();


        BruteForceParallelAlgorithm bruteForceParallel;
        pResult = bruteForceParallel.Process(problem);
        cout << fixed << setprecision(3) << "Parallel brute force duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();

        BruteForceParallelTBBAlgorithm bruteForceParallelTBB;
        pResult = bruteForceParallelTBB.Process(problem);
        cout << fixed << setprecision(3) << "Parallel brute force TBB duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();
*/
/*
        PlaneSweepAlgorithm planeSweep;
        pResult = planeSweep.Process(problem);
        cout << fixed << setprecision(3) << "Plane sweep duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();
*/
        SwitchingPlaneSweepAlgorithm switchingPlaneSweep;
        pResult = switchingPlaneSweep.Process(problem);
        cout << fixed << setprecision(3) << "Switching plane sweep duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();

        return 0;
    }
    catch(exception& ex)
    {
        cout << "Exception: " << ex.what() << endl;
        return 1;
    }
}
