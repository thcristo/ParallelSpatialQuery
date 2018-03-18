#include <iostream>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <fstream>
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
#include "PlaneSweepStripesParallelExternalAlgorithm.h"

#define NUM_ALGORITHMS 17

using namespace std;

typedef unique_ptr<AbstractAllKnnAlgorithm> algorithm_ptr_t;



int main(int argc, char* argv[])
{
    double accuracy = 1.0E-15;
    bool saveToFile = true;
    bool findDifferences = true;
    string enableAlgo(NUM_ALGORITHMS, '1');
    bool useExternalMemory = false;
    unsigned int memoryLimitMB = 0;

    if (argc < 4)
    {
        cout << "Argument error. Please enter:\n";
        cout << "Argument 1: The number of the nearest neighbors of the query,\n";
        cout << "Argument 2: The file of the input dataset,\n";
        cout << "Argument 3: The file of the training dataset,\n";
        cout << "Argument 4: The number of threads (optional)\n";
        cout << "Argument 5: The accuracy to use for comparing results (optional)\n";
        cout << "Argument 6: The number of stripes (optional)\n";
        cout << "Argument 7: Save results of each algorithm to a text file (0/1, optional)\n";
        cout << "Argument 8: Compare results of each algorithm with results of the first algorithm (0/1, optional)\n";
        cout << "Argument 9: Enable/Disable algorithms (bitstream, e.g. 01100110011110, optional)\n";
        cout << "Argument 10: Megabytes of physical memory to use for external memory algorithms (int, optional)\n";
        return 1;
    }

    try
    {
        int numNeighbors = atoi(argv[1]);
        int numThreads = 0, numStripes = 0;

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

        if (argc >= 10)
        {
            string bs = argv[9];
            if (bs.length() > 0)
            {
                if (bs.length() < NUM_ALGORITHMS)
                {
                    bs.append(NUM_ALGORITHMS - bs.length(), '0');
                }
                enableAlgo = bs;
            }
        }

        if (argc >= 11)
        {
            unsigned int limit = atoi(argv[10]);
            if (limit > 0)
            {
                memoryLimitMB = limit;
            }
        }

        vector<algorithm_ptr_t> algorithms;

        if (enableAlgo[16]== '1' || enableAlgo[17]== '1')
        {
            useExternalMemory = true;

            if (enableAlgo[16] == '1')
            {
                algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelExternalAlgorithm(numStripes, numThreads, true, false)));
            }

            if (enableAlgo[17] == '1')
            {
                algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelExternalAlgorithm(numStripes, numThreads, true, true)));
            }
        }
        else
        {
            for (int i=0; i < NUM_ALGORITHMS; ++i)
            {
                if (enableAlgo[i] == '1')
                {
                    switch(i)
                    {
                        case 0:
                            algorithms.push_back(algorithm_ptr_t(new BruteForceAlgorithm));
                            break;
                        case 1:
                            algorithms.push_back(algorithm_ptr_t(new BruteForceParallelAlgorithm(numThreads)));
                            break;
                        case 2:
                            algorithms.push_back(algorithm_ptr_t(new BruteForceParallelTBBAlgorithm(numThreads)));
                            break;
                        case 3:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepAlgorithm()));
                            break;
                        case 4:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyAlgorithm()));
                            break;
                        case 5:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelAlgorithm(numThreads, false)));
                            break;
                        case 6:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelAlgorithm(numThreads, true)));
                            break;
                        case 7:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelTBBAlgorithm(numThreads, false)));
                            break;
                        case 8:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelTBBAlgorithm(numThreads, true)));
                            break;
                        case 9:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesAlgorithm(numStripes)));
                            break;
                        case 10:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, false, false)));
                            break;
                        case 11:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, true, false)));
                            break;
                        case 12:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, false, false)));
                            break;
                        case 13:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, true, false)));
                            break;
                        case 14:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, true, true)));
                            break;
                        case 15:
                            algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, true, true)));
                            break;

                    }
                }
            }
        }

        cout.imbue(std::locale(cout.getloc(), new punct_facet<char, ',', '.'>));

        AllKnnProblem problem(argv[2], argv[3], numNeighbors, useExternalMemory, memoryLimitMB);
        unique_ptr<AllKnnResult> pResultReference, pResult;
        unique_ptr<vector<long>> pDiff;

        if (!useExternalMemory)
            cout << "Read " << problem.GetInputDataset().size() << " input points and " << problem.GetTrainingDataset().size()
                << " training points " << "in " << problem.getLoadingTime().count() << " seconds" << endl;

        auto now = chrono::system_clock::now();
        auto in_time_t = chrono::system_clock::to_time_t(now);
        stringstream ss;
        ss <<  "results_" << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << ".csv";

        ofstream outFile(ss.str(), ios_base::out);
        outFile.imbue(locale(outFile.getloc(), new punct_facet<char, ',', '.'>));

        outFile << "Algorithm;Total Duration;Sorting Duration;Total Heap Additions;Min. Heap Additions;Max. Heap Additions;Avg. Heap Additions;NumberOfStripes;Differences;First 5 different point ids" << endl;
        outFile.flush();

        for (size_t iAlgo = 0; iAlgo < algorithms.size(); ++iAlgo)
        {
            pResult = algorithms[iAlgo]->Process(problem);
            cout << fixed << setprecision(3) << algorithms[iAlgo]->GetTitle() << " duration: " << pResult->getDuration().count()
                << " sorting " << pResult->getDurationSorting().count() << " seconds,"
                << " totalAdd: " << pResult->getTotalHeapAdditions()
                <<  " minAdd: " << pResult->getMinHeapAdditions()
                << " maxAdd: " << pResult->getMaxHeapAdditions()
                << " avgAdd: " << pResult->getAvgHeapAdditions()
                << " numStripes: " << pResult->getNumStripes();

            outFile << fixed << setprecision(3) << algorithms[iAlgo]->GetTitle() << ";" << pResult->getDuration().count()
                << ";" << pResult->getDurationSorting().count()
                << ";" << pResult->getTotalHeapAdditions()
                << ";" << pResult->getMinHeapAdditions()
                << ";" << pResult->getMaxHeapAdditions()
                << ";" << pResult->getAvgHeapAdditions()
                << ";" << pResult->getNumStripes();

            if (saveToFile)
            {
                pResult->SaveToFile();
            }

            if (!useExternalMemory && findDifferences && iAlgo > 0)
            {
                pDiff = pResult->FindDifferences(*pResultReference, accuracy);
                cout << " " << pDiff->size() << " differences. ";
                outFile << ";" << pDiff->size();

                if (pDiff->size() > 0)
                {
                    cout << "First 5 different point ids: ";
                    outFile << ";";

                    for (size_t i = 0; i < 5; ++i)
                    {
                        if (pDiff->size() >= i+1)
                        {
                            cout << pDiff->at(i) << " ";
                            outFile << pDiff->at(i) << " ";
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                else
                {
                    outFile << ";";
                }
                pDiff.reset();
            }
            else
            {
                outFile << ";;";
            }

            cout << endl;
            outFile << endl;
            outFile.flush();

            if (!useExternalMemory && findDifferences && iAlgo == 0)
            {
                pResultReference = move(pResult);
            }

            pResult.reset();
        }

        outFile.close();

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
