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
#include "PlaneSweepStripesAlgorithm.h"
#include "PlaneSweepStripesParallelAlgorithm.h"
#include "PlaneSweepStripesParallelTBBAlgorithm.h"

typedef unique_ptr<AbstractAllKnnAlgorithm> algorithm_ptr_t;

int main(int argc, char* argv[])
{
    double accuracy = 1.0E-15;
    bool saveToFile = true;
    bool findDifferences = true;

    if (argc < 4)
    {
        cout << "Argument error. Please enter:\n";
        cout << "Argument 1: The number of the nearest neighbors of the query,\n";
        cout << "Argument 2: The file of the input dataset,\n";
        cout << "Argument 3: The file of the training dataset,\n";
        cout << "Argument 4: The number of threads (optional)\n";
        cout << "Argument 5: The accuracy to use for comparing results (optional)\n";
        cout << "Argument 6: The number of stripes for serial plane sweep (optional)\n";
        cout << "Argument 7: Save results of each algorithm to a text file (0/1, optional)\n";
        cout << "Argument 8: Compare results of each algorithm with results of the first algorithm (0/1, optional)\n";
        return 1;
    }

    try
    {
        int numNeighbors = atoi(argv[1]);
        int numThreads = 0, numStripes = omp_get_max_threads();

        if (argc >= 5)
        {
            int n = atoi(argv[4]);
            if (n > 0)
            {
                numThreads = n;
            }
        }

        if (argc >= 6)
        {
            double d = atof(argv[5]);
            if (d > 0.0)
            {
                accuracy = d;
            }
        }

        if (argc >= 7)
        {
            int s = atoi(argv[6]);
            if (s > 0)
            {
                numStripes = s;
            }
        }

        if (argc >= 8)
        {
            int save = atoi(argv[7]);
            if (save == 0)
            {
                saveToFile = false;
            }
        }

        if (argc >= 9)
        {
            int compare = atoi(argv[8]);
            if (compare == 0)
            {
                findDifferences = false;
            }
        }

        AllKnnProblem problem(argv[2], argv[3], numNeighbors);
        unique_ptr<AllKnnResult> pResultReference, pResult;
        unique_ptr<vector<long>> pDiff;
        cout << "Read " << problem.GetInputDataset().size() << " input points and " << problem.GetTrainingDataset().size() << " training points." << endl;

        vector<algorithm_ptr_t> algorithms;

        algorithms.push_back(algorithm_ptr_t(new BruteForceParallelAlgorithm(numThreads)));
        algorithms.push_back(algorithm_ptr_t(new BruteForceParallelTBBAlgorithm(numThreads)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepAlgorithm()));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyAlgorithm()));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelAlgorithm(numThreads, false)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelAlgorithm(numThreads, true)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelTBBAlgorithm(numThreads, false)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelTBBAlgorithm(numThreads, true)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesAlgorithm(numStripes)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numThreads, false)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numThreads, true)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numThreads, false)));
        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numThreads, true)));

        for (auto algo = algorithms.cbegin(); algo < algorithms.cend(); ++algo)
        {
            pResult = (*algo)->Process(problem);
            cout << fixed << setprecision(3) << (*algo)->GetTitle() << " duration: " << pResult->getDuration().count()
                << " sorting " << pResult->getDurationSorting().count() << " seconds" ;

            if (saveToFile)
            {
                pResult->SaveToFile();
            }

            if (findDifferences && algo != algorithms.cbegin())
            {
                pDiff = pResult->FindDifferences(*pResultReference, accuracy);
                cout << " " << pDiff->size() << " differences. ";
                if (pDiff->size() > 0)
                {
                    cout << "First 5 different point ids: ";
                    for (size_t i = 0; i < 5; ++i)
                    {
                        if (pDiff->size() >= i+1)
                        {
                            cout << pDiff->at(i) << " ";
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                pDiff.reset();
            }

            cout << endl;

            if (findDifferences && algo == algorithms.cbegin())
            {
                pResultReference = move(pResult);
            }

            pResult.reset();
        }

        if (pResultReference)
        {
            pResultReference.reset();
        }

        return 0;
    }
    catch(exception& ex)
    {
        cout << "Exception: " << ex.what() << endl;
        return 1;
    }
}
