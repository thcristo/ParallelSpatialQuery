#include <iostream>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "BruteForceAlgorithm.h"
#include "BruteForceParallelAlgorithm.h"
#include "BruteForceParallelTBBAlgorithm.h"
#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "PlaneSweepAlgorithm.h"
#include "PlaneSweepCopyAlgorithm.h"
#include "PlaneSweepCopyParallelAlgorithm.h"
#include "PlaneSweepCopyParallelTBBAlgorithm.h"
#include "PlaneSweepFullCopyParallelAlgorithm.h"
#include "PlaneSweepStripesParallelAlgorithm.h"

int main(int argc, char* argv[])
{

    if (argc < 4)
    {
        cout << "Argument error. Please enter:\n";
        cout << "Argument 1: The number of the nearest neighbors of the query,\n";
        cout << "Argument 2: The file of the input dataset,\n";
        cout << "Argument 3: The file of the training dataset,\n";
        cout << "Argument 4: The number of threads (optional)\n";
        return 1;
    }

    try
    {
        int numNeighbors = atoi(argv[1]);
        int numThreads = 0;

        if (argc >= 5)
        {
            int n = atoi(argv[4]);
            if (n > 0)
            {
                numThreads = n;
            }
        }

        AllKnnProblem problem(argv[2], argv[3], numNeighbors);

        cout << "Read " << problem.GetInputDataset().size() << " input points and " << problem.GetTrainingDataset().size() << " training points." << endl;
        unique_ptr<AllKnnResult> pResult;

/*
        BruteForceAlgorithm bruteForce;
        pResult = bruteForce.Process(problem);
        cout << fixed << setprecision(3) << "Brute force duration: " << pResult->duration().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();
*/

        BruteForceParallelAlgorithm bruteForceParallel(numThreads);
        pResult = bruteForceParallel.Process(problem);
        cout << fixed << setprecision(3) << "Parallel brute force duration: " << pResult->duration().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();

        BruteForceParallelTBBAlgorithm bruteForceParallelTBB(numThreads);
        pResult = bruteForceParallelTBB.Process(problem);
        cout << fixed << setprecision(3) << "Parallel brute force TBB duration: " << pResult->duration().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();


        PlaneSweepAlgorithm planeSweep;
        pResult = planeSweep.Process(problem);
        cout << fixed << setprecision(3) << "Plane sweep duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();

        PlaneSweepCopyAlgorithm planeSweepCopy;
        pResult = planeSweepCopy.Process(problem);
        cout << fixed << setprecision(3) << "Plane sweep copy duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();
        problem.ClearSortedVectors();

        PlaneSweepCopyParallelAlgorithm planeSweepCopyParallel(numThreads);
        pResult = planeSweepCopyParallel.Process(problem);
        cout << fixed << setprecision(3) << "Parallel plane sweep copy duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();
        problem.ClearSortedVectors();

        PlaneSweepCopyParallelTBBAlgorithm planeSweepCopyParallelTBB(numThreads);
        pResult = planeSweepCopyParallelTBB.Process(problem);
        cout << fixed << setprecision(3) << "Parallel plane sweep copy TBB duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();
        problem.ClearSortedVectors();

        PlaneSweepStripesParallelAlgorithm planeSweepStripesParallel(numThreads);
        pResult = planeSweepStripesParallel.Process(problem);
        cout << fixed << setprecision(3) << "Parallel plane sweep with stripes duration: " << pResult->duration().count() << " sorting " << pResult->durationSorting().count() << " seconds" << endl;
        pResult->SaveToFile();
        pResult.reset();
        problem.ClearSortedVectors();

        return 0;
    }
    catch(exception& ex)
    {
        cout << "Exception: " << ex.what() << endl;
        return 1;
    }
}
