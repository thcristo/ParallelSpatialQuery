/* This file contains the main function of the program */

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
#include "PlaneSweepStripesParallelExternalTBBAlgorithm.h"

#define NUM_ALGORITHMS 30

using namespace std;

typedef unique_ptr<AbstractAllKnnAlgorithm> algorithm_ptr_t;

int main(int argc, char* argv[])
{
    double accuracy = 1.0E-15;
    bool saveToFile = true;
    bool findDifferences = true;
    string enableAlgo(NUM_ALGORITHMS, '1');
    bool useExternalMemory = false;
    bool useInternalMemory = false;
    size_t memoryLimitMB = 1024;

    //parameters must be specified in the command line
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
        cout << "Argument 9: Enable/Disable algorithms (bitstream of 30 digits 0 or 1, e.g. 01100110011110, optional)\n";
        cout << "Argument 10: Megabytes of physical memory to use for external memory algorithms (int, optional)\n";
        return 1;
    }

    try
    {
        int numNeighbors = atoi(argv[1]);
        int numThreads = 0, numStripes = 0;

        //set number of threads to use
        if (argc >= 5)
        {
            int n = atoi(argv[4]);
            if (n > 0)
            {
                numThreads = n;
            }
        }

        //set accuracy for comparing results
        if (argc >= 6)
        {
            double d = atof(argv[5]);
            if (d > 0.0)
            {
                accuracy = d;
            }
        }

        //set number of stripes, if numStripes=0 then it will be calculated automatically
        if (argc >= 7)
        {
            int s = atoi(argv[6]);
            if (s > 0)
            {
                numStripes = s;
            }
        }

        //set if we want to save the list of neighbors in a text file
        if (argc >= 8)
        {
            int save = atoi(argv[7]);
            if (save == 0)
            {
                saveToFile = false;
            }
        }

        //set if we want to check for differences between results using the specified accuracy
        if (argc >= 9)
        {
            int compare = atoi(argv[8]);
            if (compare == 0)
            {
                findDifferences = false;
            }
        }

        //the bitstream of algorithms to run, a sequence of 30 digits 0 or 1
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

        //memory limit in MB for the external memory algorithms, do not use more memory
        if (argc >= 11)
        {
            size_t limit = stoull(argv[10]);
            if (limit > 0)
            {
                memoryLimitMB = limit;
            }
        }

        vector<algorithm_ptr_t> algorithms;

        //insert all algorithms we want to run in a vector
        for (int i=0; i < NUM_ALGORITHMS; ++i)
        {
            if (enableAlgo[i] == '1')
            {
                switch(i)
                {
                    case 0:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new BruteForceAlgorithm));
                        break;
                    case 1:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new BruteForceParallelAlgorithm(numThreads)));
                        break;
                    case 2:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new BruteForceParallelTBBAlgorithm(numThreads)));
                        break;
                    case 3:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepAlgorithm()));
                        break;
                    case 4:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyAlgorithm()));
                        break;
                    case 5:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelAlgorithm(numThreads, false)));
                        break;
                    case 6:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelAlgorithm(numThreads, true)));
                        break;
                    case 7:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelTBBAlgorithm(numThreads, false)));
                        break;
                    case 8:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepCopyParallelTBBAlgorithm(numThreads, true)));
                        break;
                    case 9:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesAlgorithm(numStripes)));
                        break;

                    case 10:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, false, false, false)));
                        break;
                    case 11:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, false, true, false)));
                        break;
                    case 12:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, true, false, false)));
                        break;
                    case 13:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, true, true, false)));
                        break;
                    case 14:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, false, false, false)));
                        break;
                    case 15:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, false, true, false)));
                        break;
                    case 16:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, true, false, false)));
                        break;
                    case 17:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, true, true, false)));
                        break;

                    case 18:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, false, false, true)));
                        break;
                    case 19:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, false, true, true)));
                        break;
                    case 20:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, true, false, true)));
                        break;
                    case 21:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelAlgorithm(numStripes, numThreads, true, true, true)));
                        break;
                    case 22:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, false, false, true)));
                        break;
                    case 23:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, false, true, true)));
                        break;
                    case 24:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, true, false, true)));
                        break;
                    case 25:
                        useInternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelTBBAlgorithm(numStripes, numThreads, true, true, true)));
                        break;

                    case 26:
                        useExternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelExternalAlgorithm(numStripes, numThreads, true, false)));
                        break;
                    case 27:
                        useExternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelExternalAlgorithm(numStripes, numThreads, true, true)));
                        break;
                    case 28:
                        useExternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelExternalTBBAlgorithm(numStripes, numThreads, true, false)));
                        break;
                    case 29:
                        useExternalMemory = true;
                        algorithms.push_back(algorithm_ptr_t(new PlaneSweepStripesParallelExternalTBBAlgorithm(numStripes, numThreads, true, true)));
                        break;
                }
            }
        }

        //set Greek numeric formatting for decimal and thousand separator
        cout.imbue(std::locale(cout.getloc(), new punct_facet<char, ',', '.'>));

        unique_ptr<AllKnnProblem> pProblem;
        unique_ptr<AllKnnProblemExternal> pProblemExternal;

        //allocate the problem object depending on which kind of algorithm we need to run, internal memory or external, may be both of them
        if (useInternalMemory)
            pProblem.reset(new AllKnnProblem(argv[2], argv[3], numNeighbors, true));

        if (useExternalMemory)
            pProblemExternal.reset(new AllKnnProblemExternal(argv[2], argv[3], numNeighbors, true, memoryLimitMB));

        unique_ptr<AllKnnResult> pResultReference, pResult;
        unique_ptr<vector<unsigned long>> pDiff;

        //report how much time was required for loading the datasets, input and training
        if (useInternalMemory)
            cout << "Read " << pProblem->GetInputDatasetSize() << " input points and " << pProblem->GetTrainingDatasetSize()
                << " training points " << "in " << pProblem->getLoadingTime().count() << " seconds" << endl;

        if (useExternalMemory)
            cout << "Read " << pProblemExternal->GetInputDatasetSize() << " input points and " << pProblemExternal->GetTrainingDatasetSize()
                << " training points " << "in " << pProblemExternal->getLoadingTime().count() << " seconds" << endl;

        //create the output file to record performance statistics
        auto now = chrono::system_clock::now();
        auto in_time_t = chrono::system_clock::to_time_t(now);
        stringstream ss;
        ss <<  "results_" << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << ".csv";

        ofstream outFile(ss.str(), ios_base::out);
        outFile.imbue(locale(outFile.getloc(), new punct_facet<char, ',', '.'>));

        outFile << "Algorithm;Total Duration;Sorting Duration;Total Heap Additions;Min. Heap Additions;Max. Heap Additions;Avg. Heap Additions;NumberOfStripes;HasAllocationError;PendingPoints;NumFirstPassWindows;NumSecondPassWindows;CommitWindow Duration;Final Sorting Duration;Differences;First 5 different point ids" << endl;
        outFile.flush();

        //run each requested algorithm
        for (size_t iAlgo = 0; iAlgo < algorithms.size(); ++iAlgo)
        {
            //process the correct type of problem (external or internal memory)
            if (algorithms[iAlgo]->UsesExternalMemory())
                pResult = algorithms[iAlgo]->Process(*pProblemExternal);
            else
                pResult = algorithms[iAlgo]->Process(*pProblem);

            //output the performance statistics to the console
            cout << fixed << setprecision(3) << algorithms[iAlgo]->GetTitle() << " duration: " << pResult->getDuration().count()
                << " sorting " << pResult->getDurationSorting().count() << " seconds "
                << " totalAdd: " << pResult->getTotalHeapAdditions()
                <<  " minAdd: " << pResult->getMinHeapAdditions()
                << " maxAdd: " << pResult->getMaxHeapAdditions()
                << " avgAdd: " << pResult->getAvgHeapAdditions()
                << " numStripes: " << pResult->getNumStripes()
                << " hasAllocationError: " << pResult->HasAllocationError()
                << " numPendingPoints: " << pResult->getNumPendingPoints()
                << " numFirstPassWindows: " << pResult->getNumFirstPassWindows()
                << " numSecondPassWindows: " << pResult->getNumSecondPassWindows()
                << " commitWindow: " << pResult->getDurationCommitWindow().count() << " seconds "
                << " finalSorting: " << pResult->getDurationFinalSorting().count() << " seconds ";

            //write the performance statistics to the output file
            outFile << fixed << setprecision(3) << algorithms[iAlgo]->GetTitle() << ";" << pResult->getDuration().count()
                << ";" << pResult->getDurationSorting().count()
                << ";" << pResult->getTotalHeapAdditions()
                << ";" << pResult->getMinHeapAdditions()
                << ";" << pResult->getMaxHeapAdditions()
                << ";" << pResult->getAvgHeapAdditions()
                << ";" << pResult->getNumStripes()
                << ";" << pResult->HasAllocationError()
                << ";" << pResult->getNumPendingPoints()
                << ";" << pResult->getNumFirstPassWindows()
                << ";" << pResult->getNumSecondPassWindows()
                << ";" << pResult->getDurationCommitWindow().count()
                << ";" << pResult->getDurationFinalSorting().count();

            //save the list of neighbors to a text file
            if (saveToFile && !pResult->HasAllocationError())
            {
                pResult->SaveToFile();
            }

            //check for differences between distances of neighbors by using the first algorithm as a reference result
            if (findDifferences && iAlgo > 0 && !pResult->HasAllocationError())
            {
                pDiff = pResult->FindDifferences(*pResultReference, accuracy);
                cout << " " << pDiff->size() << " differences. ";
                outFile << ";" << pDiff->size();

                if (pDiff->size() > 0)
                {
                    //report the first 5 different neighbor ids
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

            //if we need to check for differences, keep the first result to use as a reference for comparing the others with it
            if (findDifferences && iAlgo == 0)
            {
                pResultReference = move(pResult);
            }

            //free the memory allocated for the result
            pResult.reset();
        }

        outFile.close();

        //we are exiting, free the reference result
        if (pResultReference)
        {
            pResultReference.reset();
        }

        return 0;
    }
    catch(exception& ex)
    {
        //report any exception
        cout << "Exception: " << ex.what() << endl;
        return 1;
    }

}
