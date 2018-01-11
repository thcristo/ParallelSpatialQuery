#include <iostream>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <chrono>
#include <memory>
#include <omp.h>
#include <fstream>
#include "BruteForceAlgorithm.h"
#include "BruteForceParallelAlgorithm.h"
#include "BruteForceParallelTBBAlgorithm.h"
#include "AllKnnProblem.h"
#include "AllKnnResult.h"
#include "AllKnnResultSorted.h"
#include "AllKnnResultStripes.h"
#include "PlaneSweepAlgorithm.h"
#include "PlaneSweepCopyAlgorithm.h"
#include "PlaneSweepCopyParallelAlgorithm.h"
#include "PlaneSweepCopyParallelTBBAlgorithm.h"
#include "PlaneSweepStripesAlgorithm.h"
#include "PlaneSweepStripesParallelAlgorithm.h"
#include "PlaneSweepStripesParallelTBBAlgorithm.h"

#define NUM_ALGORITHMS 14

using namespace std;

template<class ProblemT, class ResultT, class PointVectorT, class PointVectorIteratorT, class NeighborVectorT>
using algorithm_ptr_t = unique_ptr<AbstractAllKnnAlgorithm<ProblemT, ResultT, PointVectorT, PointVectorIteratorT, NeighborVectorT>>;



template <class charT, charT decimalSeparator, charT thousandsSeparator>
class punct_facet: public numpunct<charT> {
protected:
    charT do_decimal_point() const { return decimalSeparator; }
    charT do_thousands_sep() const { return thousandsSeparator; }
    string do_grouping() const { return "\03"; }
};

typedef AllKnnProblem<point_vector_t> AllKnnProblemMem;
typedef AllKnnResult<AllKnnProblemMem, pointNeighbors_priority_queue_vector_t, point_vector_t, vector<long>, point_vector_iterator_t> AllKnnResultMem;
typedef AllKnnResultSorted<AllKnnProblemMem, pointNeighbors_priority_queue_vector_t, point_vector_t, vector<long>, point_vector_iterator_t> AllKnnResultSortedMem;
typedef algorithm_ptr_t<AllKnnProblemMem, AllKnnResultMem, point_vector_t, point_vector_iterator_t, pointNeighbors_priority_queue_vector_t> algorithm_ptr_mem_t;
typedef StripeData<point_vector_vector_t, vector<StripeBoundaries_t>> StripeDataMem;
typedef AllKnnResultStripes<AllKnnProblemMem, pointNeighbors_priority_queue_vector_t, point_vector_t, vector<long>, point_vector_iterator_t, point_vector_vector_t, vector<StripeBoundaries_t>> AllKnnResultStripesMem;

typedef AllKnnProblem<point_vector_ext_t> AllKnnProblemExt;
//typedef AllKnnResult<AllKnnProblemExt, pointNeighbors_priority_queue_vector_t, point_vector_ext_t, ext_vector<long>, point_vector_iterator_ext_t> AllKnnResultExt;

template<class ProblemT, class ResultBaseT, class PointVectorT, class PointVectorIteratorT, class PointIdVectorT, class NeighborVectorT>
void RunAlgorithms(ProblemT& problem, vector<algorithm_ptr_t<ProblemT, ResultBaseT, PointVectorT, PointVectorIteratorT, NeighborVectorT>>& algorithms,
                   ofstream& outFile, bool saveToFile, bool findDifferences, double accuracy)
{
    unique_ptr<ResultBaseT> pResultReference, pResult;
    unique_ptr<PointIdVectorT> pDiff;

    cout << "Read " << problem.GetInputDataset().size() << " input points and " << problem.GetTrainingDataset().size()
            << " training points " << "in " << problem.getLoadingTime().count() << " seconds" << endl;

    for (size_t iAlgo = 0; iAlgo < algorithms.size(); ++iAlgo)
    {
        pResult = algorithms[iAlgo]->Process(problem);
        cout << fixed << setprecision(3) << algorithms[iAlgo]->GetTitle() << " duration: " << pResult->getDuration().count()
            << " sorting " << pResult->getDurationSorting().count() << " seconds,"
            << " totalAdd: " << pResult->getTotalHeapAdditions()
            <<  " minAdd: " << pResult->getMinHeapAdditions()
            << " maxAdd: " << pResult->getMaxHeapAdditions()
            << " avgAdd: " << pResult->getAvgHeapAdditions();

        outFile << fixed << setprecision(3) << algorithms[iAlgo]->GetTitle() << ";" << pResult->getDuration().count()
            << ";" << pResult->getDurationSorting().count()
            << ";" << pResult->getTotalHeapAdditions()
            << ";" << pResult->getMinHeapAdditions()
            << ";" << pResult->getMaxHeapAdditions()
            << ";" << pResult->getAvgHeapAdditions();

        if (saveToFile)
        {
            pResult->SaveToFile();
        }

        if (findDifferences && iAlgo > 0)
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

        if (findDifferences && iAlgo == 0)
        {
            pResultReference = move(pResult);
        }

        pResult.reset();
    }

    if (pResultReference)
    {
        pResultReference.reset();
    }
}

int main(int argc, char* argv[])
{
    double accuracy = 1.0E-15;
    bool saveToFile = true;
    bool findDifferences = true;
    string enableAlgo(NUM_ALGORITHMS, '1');
    bool useExternalMemory = false;

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
        cout << "Argument 10: Use STXXL (0/1, optional)\n";
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

        if (argc >=11)
        {
            int xxl = atoi(argv[10]);
            if (xxl != 0)
            {
                useExternalMemory = true;
            }
        }

        cout.imbue(std::locale(cout.getloc(), new punct_facet<char, ',', '.'>));

        auto now = chrono::system_clock::now();
        auto in_time_t = chrono::system_clock::to_time_t(now);
        stringstream ss;
        ss <<  "results_" << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << ".csv";

        ofstream outFile(ss.str(), ios_base::out);
        outFile.imbue(locale(outFile.getloc(), new punct_facet<char, ',', '.'>));

        outFile << "Algorithm;Total Duration;Sorting Duration;Total Heap Additions;Min. Heap Additions;Max. Heap Additions;Avg. Heap Additions;Differences;First 5 different point ids" << endl;
        outFile.flush();

        if (useExternalMemory)
        {
            AllKnnProblemExt problem(argv[2], argv[3], numNeighbors);
        }
        else
        {
            AllKnnProblemMem problem(argv[2], argv[3], numNeighbors);

            vector<algorithm_ptr_mem_t> algorithms;

            for (int i=0; i < NUM_ALGORITHMS; ++i)
            {
                if (enableAlgo[i] == '1')
                {
                    switch(i)
                    {
                        case 0:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new BruteForceAlgorithm<AllKnnProblemMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t>()));
                            break;
                        case 1:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new BruteForceParallelAlgorithm<AllKnnProblemMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t>(numThreads)));
                            break;
                        case 2:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new BruteForceParallelTBBAlgorithm<AllKnnProblemMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t>(numThreads)));
                            break;
                        case 3:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepAlgorithm<AllKnnProblemMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t, point_vector_index_t>()));
                            break;
                        case 4:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepCopyAlgorithm<AllKnnProblemMem, AllKnnResultSortedMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t>()));
                            break;
                        case 5:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepCopyParallelAlgorithm<AllKnnProblemMem, AllKnnResultSortedMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t>(numThreads, false)));
                            break;
                        case 6:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepCopyParallelAlgorithm<AllKnnProblemMem, AllKnnResultSortedMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t>(numThreads, true)));
                            break;
                        case 7:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepCopyParallelTBBAlgorithm<AllKnnProblemMem, AllKnnResultSortedMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t>(numThreads, false)));
                            break;
                        case 8:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepCopyParallelTBBAlgorithm<AllKnnProblemMem, AllKnnResultSortedMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t>(numThreads, true)));
                            break;
                        case 9:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepStripesAlgorithm<AllKnnProblemMem, AllKnnResultStripesMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t, StripeDataMem>(numStripes)));
                            break;
                        case 10:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepStripesParallelAlgorithm<AllKnnProblemMem, AllKnnResultStripesMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t, StripeDataMem>(numStripes, numThreads, false)));
                            break;
                        case 11:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepStripesParallelAlgorithm<AllKnnProblemMem, AllKnnResultStripesMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t, StripeDataMem>(numStripes, numThreads, true)));
                            break;
                        case 12:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepStripesParallelTBBAlgorithm<AllKnnProblemMem, AllKnnResultStripesMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t, StripeDataMem>(numStripes, numThreads, false)));
                            break;
                        case 13:
                            algorithms.push_back(algorithm_ptr_mem_t(
                                new PlaneSweepStripesParallelTBBAlgorithm<AllKnnProblemMem, AllKnnResultStripesMem, AllKnnResultMem, point_vector_t,
                                    point_vector_iterator_t, pointNeighbors_priority_queue_vector_t, StripeDataMem>(numStripes, numThreads, true)));
                            break;
                    }
                }
            }

            RunAlgorithms<AllKnnProblemMem, AllKnnResultMem, point_vector_t, point_vector_iterator_t, vector<long>, pointNeighbors_priority_queue_vector_t>(problem, algorithms, outFile, saveToFile, findDifferences, accuracy);
        }

        outFile.close();

        return 0;
    }
    catch(exception& ex)
    {
        cout << "Exception: " << ex.what() << endl;
        return 1;
    }
}


