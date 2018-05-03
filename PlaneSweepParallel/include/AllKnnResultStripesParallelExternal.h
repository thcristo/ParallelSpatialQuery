#ifndef ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
#define ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
#include <stxxl/sort>
#include "AllKnnResultStripes.h"
#include "StripesWindow.h"
#include "AllKnnProblemExternal.h"

struct ExternalPointComparerY
{
    Point minval = {0, 0.0, -0.00000001};
    Point maxval = {0, 0.0, 1.00000001};

    bool operator()(const Point& point1, const Point& point2) const
    {
        return point1.y < point2.y;
    }

    const Point& min_value() const
    {
        return minval;
    }

    const Point& max_value() const
    {
        return maxval;
    }
};

struct ExternalPointComparerX
{
    Point minval = {0, -0.00000001, 0.0};
    Point maxval = {0, 1.00000001, 0.0};

    bool operator()(const Point& point1, const Point& point2) const
    {
        return point1.x < point2.x;
    }

    const Point& min_value() const
    {
        return minval;
    }

    const Point& max_value() const
    {
        return maxval;
    }
};

struct ExternalNeighborComparer
{
    NeighborExt minval = {0, 0.0, 0, 0};
    NeighborExt maxval = {0, 0.0, ULONG_MAX, UINT_MAX};

    bool operator()(const NeighborExt& neighbor1, const NeighborExt& neighbor2) const
    {
        return neighbor1.inputPointId < neighbor2.inputPointId || ( (neighbor1.inputPointId == neighbor2.inputPointId) && (neighbor1.position < neighbor2.position) );
    }

    const NeighborExt& min_value() const
    {
        return minval;
    }

    const NeighborExt& max_value() const
    {
        return maxval;
    }
};

class AllKnnResultStripesParallelExternal : public AllKnnResult
{
    public:
        AllKnnResultStripesParallelExternal(const AllKnnProblemExternal& problem, const string& filePrefix)
            : AllKnnResult(problem, filePrefix), problemExt(problem)
        {
        }

        AllKnnResultStripesParallelExternal(const AllKnnProblemExternal& problem, const string& filePrefix, bool parallelSort, bool splitByT)
            : AllKnnResult(problem, filePrefix), splitByT(splitByT), parallelSort(parallelSort), problemExt(problem)
        {
        }

        virtual ~AllKnnResultStripesParallelExternal()
        {
        }

        size_t SplitStripes(size_t numStripes)
        {
            ext_point_vector_t inputDatasetSortedY(problemExt.GetExtInputDataset());
            ext_point_vector_t trainingDatasetSortedY(problemExt.GetExtTrainingDataset());

            auto memoryLimit = problemExt.GetMemoryLimitBytes();

            stxxl::sort(inputDatasetSortedY.cbegin(), inputDatasetSortedY.cend(), ExternalPointComparerY(), memoryLimit);
            stxxl::sort(trainingDatasetSortedY.cbegin(), trainingDatasetSortedY.cend(), ExternalPointComparerY(), memoryLimit);

            if (numStripes > 0)
            {
                create_fixed_stripes(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            }
            else
            {
                numStripes = get_optimal_stripes();
                create_fixed_stripes(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            }

            return pStripeBoundaries->size();
        }

        unique_ptr<StripesWindow> GetWindow(size_t fromStripe, bool secondPass)
        {
            auto memoryLimit = problemExt.GetMemoryLimitBytes();
            auto safeMemoryLimit = 9*memoryLimit/10;

            size_t numNeighbors = problem.GetNumNeighbors();
            size_t numStripes = pStripeBoundaries->size();
            size_t usedMemory = (pPendingPoints->size()*(sizeof(unsigned long) + sizeof(StripePoint)))
                                + (pPendingNeighborsContainer->size()*(sizeof(unsigned long) + sizeof(PointNeighbors<neighbors_priority_queue_t>) + numNeighbors*sizeof(Neighbor)))
                                + (pPendingPoints->size()*(sizeof(Point)))
                                + 2*(pPendingPoints->size()*(2*sizeof(void*)))
                                + 4*numStripes*sizeof(size_t)
                                + numStripes*sizeof(StripeBoundaries_t)
                                + 6*64*1024*1024;

            cout << "pending points " << pPendingPoints->size() << " " << pPendingNeighborsContainer->size() << endl;
            cout << "reserved memory " << usedMemory << endl;

            if (secondPass)
            {
                size_t endStripe = fromStripe;
                size_t startStripe = endStripe + 1;

                do
                {
                    size_t sizeTraining = (pTrainingStripeCount->at(startStripe - 1))*sizeof(Point);

                    if (usedMemory + sizeTraining <= safeMemoryLimit)
                    {
                        usedMemory += sizeTraining;
                        --startStripe;
                    }
                    else
                        break;
                } while (startStripe > 0);

                if (endStripe < startStripe)
                {
                    hasAllocationError = true;
                    return unique_ptr<StripesWindow>(nullptr);
                }

                size_t numWindowStripes = endStripe - startStripe + 1;
                unique_ptr<point_vector_vector_t> pTrainingStripes(new point_vector_vector_t(numWindowStripes));
                unique_ptr<vector<StripeBoundaries_t>> pBoundaries(new vector<StripeBoundaries_t>(numWindowStripes));

                for (size_t iStripe = startStripe; iStripe <= endStripe; ++iStripe)
                {
                    auto& trainingPoints = pTrainingStripes->at(iStripe-startStripe);
                    auto& boundaries = pBoundaries->at(iStripe-startStripe);

                    if (pTrainingStripeCount->at(iStripe) > 0)
                    {
                        auto trainingStart = pStripedTrainingDataset->cbegin() + pTrainingStripeOffset->at(iStripe);
                        auto trainingEnd = trainingStart + pTrainingStripeCount->at(iStripe);
                        trainingPoints.reserve(pTrainingStripeCount->at(iStripe));
                        std::copy(trainingStart, trainingEnd, std::back_inserter(trainingPoints));
                    }

                    boundaries.minY = pStripeBoundaries->at(iStripe).minY;
                    boundaries.maxY = pStripeBoundaries->at(iStripe).maxY;
                }

                unique_ptr<StripesWindow> pWindow(new StripesWindow(startStripe, endStripe, pTrainingStripes, pBoundaries));
                ++numSecondPassWindows;
                return pWindow;
            }
            else
            {
                size_t startStripe = fromStripe;
                size_t endStripe = startStripe;

                do
                {
                    size_t sizeInput = (pInputStripeCount->at(endStripe))*sizeof(Point);
                    size_t sizeTraining = (pTrainingStripeCount->at(endStripe))*sizeof(Point);
                    size_t sizeNeighbors = (pInputStripeCount->at(endStripe))*(sizeof(PointNeighbors<neighbors_priority_queue_t>) + numNeighbors*sizeof(Neighbor));
                    size_t additionalMemory = sizeInput + sizeTraining + sizeNeighbors;

                    if (usedMemory + additionalMemory <= safeMemoryLimit)
                    {
                        usedMemory += additionalMemory;
                        ++endStripe;
                    }
                    else
                        break;
                } while (endStripe <= numStripes - 1);

                if (endStripe <= startStripe)
                {
                    hasAllocationError = true;
                    return unique_ptr<StripesWindow>(nullptr);
                }

                size_t numWindowStripes = endStripe - startStripe;

                unique_ptr<point_vector_vector_t> pInputStripes(new point_vector_vector_t(numWindowStripes)), pTrainingStripes(new point_vector_vector_t(numWindowStripes));
                unique_ptr<vector<StripeBoundaries_t>> pBoundaries(new vector<StripeBoundaries_t>(numWindowStripes));

                for (size_t iStripe = startStripe; iStripe < endStripe; ++iStripe)
                {
                    auto& inputPoints = pInputStripes->at(iStripe-startStripe);
                    auto& trainingPoints = pTrainingStripes->at(iStripe-startStripe);
                    auto& boundaries = pBoundaries->at(iStripe-startStripe);

                    if (pInputStripeCount->at(iStripe) > 0)
                    {
                        auto inputStart = pStripedInputDataset->cbegin() + pInputStripeOffset->at(iStripe);
                        auto inputEnd = inputStart + pInputStripeCount->at(iStripe);
                        inputPoints.reserve(pInputStripeCount->at(iStripe));
                        std::copy(inputStart, inputEnd, std::back_inserter(inputPoints));
                    }

                    if (pTrainingStripeCount->at(iStripe) > 0)
                    {
                        auto trainingStart = pStripedTrainingDataset->cbegin() + pTrainingStripeOffset->at(iStripe);
                        auto trainingEnd = trainingStart + pTrainingStripeCount->at(iStripe);
                        trainingPoints.reserve(pTrainingStripeCount->at(iStripe));
                        std::copy(trainingStart, trainingEnd, std::back_inserter(trainingPoints));
                    }

                    boundaries.minY = pStripeBoundaries->at(iStripe).minY;
                    boundaries.maxY = pStripeBoundaries->at(iStripe).maxY;
                }

                unique_ptr<StripesWindow> pWindow(new StripesWindow(startStripe, endStripe-1, pInputStripes, pTrainingStripes, pBoundaries, numNeighbors));
                ++numFirstPassWindows;
                return pWindow;
            }
        }

        bool HasAllocationError() override
        {
            return hasAllocationError;
        }

        unique_ptr<point_vector_t> GetPendingPointsForWindow(const StripesWindow& window)
        {
            unique_ptr<point_vector_t> pPoints(new point_vector_t());
            pPoints->reserve(pPendingPoints->size());
            auto pendingPointsBegin = pPendingPoints->cbegin();
            auto pendingPointsEnd = pPendingPoints->cend();

            if (window.IsSecondPass())
            {
                size_t startStripe = window.GetStartStripe();

                for (auto pendingIter = pendingPointsBegin; pendingIter != pendingPointsEnd; ++pendingIter)
                {
                    auto pointId = pendingIter->second.id;
                    auto& neighbors = pPendingNeighborsContainer->at(pointId);
                    size_t lowStripe = neighbors.getLowStripe();

                    if (lowStripe > startStripe)
                    {
                        pPoints->push_back({pendingIter->second.id, pendingIter->second.x, pendingIter->second.y});
                    }
                }
            }
            else
            {
                size_t endStripe = window.GetEndStripe();

                for (auto pendingIter = pendingPointsBegin; pendingIter != pendingPointsEnd; ++pendingIter)
                {
                    auto pointId = pendingIter->second.id;
                    auto& neighbors = pPendingNeighborsContainer->at(pointId);
                    size_t highStripe = neighbors.getHighStripe();

                    if (highStripe < endStripe)
                    {
                        pPoints->push_back({pointId, pendingIter->second.x, pendingIter->second.y});
                    }
                }
            }

            return pPoints;
        }

        pointNeighbors_priority_queue_map_t& GetPendingNeighborsContainer()
        {
            return *pPendingNeighborsContainer;
        }

        void CommitWindow(StripesWindow& window, point_vector_t& pendingPoints)
        {
            auto commitStart = chrono::high_resolution_clock::now();

            if (pNeighborsExtVector == nullptr)
            {
                pNeighborsExtVector.reset(new ext_neighbors_vector_t());
                pNeighborsExtVector->reserve(problem.GetNumNeighbors()*problem.GetInputDatasetSize());
            }

            if (pHeapAdditionsVector == nullptr)
            {
                pHeapAdditionsVector.reset(new ext_size_vector_t());
                pHeapAdditionsVector->reserve(problem.GetInputDatasetSize());
            }

            auto pendingPointsBegin = pendingPoints.cbegin();
            auto pendingPointsEnd = pendingPoints.cend();

            for (auto pendingPointsIter = pendingPointsBegin; pendingPointsIter < pendingPointsEnd; ++pendingPointsIter)
            {
                auto pointId = pendingPointsIter->id;
                auto& pointNeighbors = pPendingNeighborsContainer->at(pointId);
                if (IsSearchCompleted(pointNeighbors))
                {
                    pHeapAdditionsVector->push_back(pointNeighbors.GetNumAdditions());

                    unsigned int neighborPosition = 0;

                    while (pointNeighbors.HasNext())
                    {
                        Neighbor neighbor = pointNeighbors.Next();

                        NeighborExt extNeighbor = {neighbor.pointId, neighbor.distanceSquared, pointId, neighborPosition};

                        pNeighborsExtVector->push_back(extNeighbor);
                        ++neighborPosition;
                    }
                    pPendingPoints->erase(pointId);
                    pPendingNeighborsContainer->erase(pointId);
                }
            }

            bool isSecondPass = window.IsSecondPass();
            if (!isSecondPass)
            {
                auto stripeData = window.GetStripeData();
                auto& neighborsContainer = window.GetNeighborsContainer();
                size_t numWindowStripes = window.GetNumStripes();
                size_t windowStartStripe = window.GetStartStripe();

                for (size_t iWindowStripe = 0; iWindowStripe < numWindowStripes; ++iWindowStripe)
                {
                    auto& inputDataset = stripeData.InputDatasetStripe[iWindowStripe];
                    size_t numInputPoints = inputDataset.size();

                    if (numInputPoints > 0)
                    {
                        auto& stripeNeighbors = neighborsContainer.at(iWindowStripe);

                        for (size_t iPoint=0; iPoint < numInputPoints; ++iPoint)
                        {
                            auto& pointNeighbors = stripeNeighbors[iPoint];
                            auto& point = inputDataset[iPoint];
                            unsigned long pointId = point.id;

                            if (IsSearchCompleted(pointNeighbors))
                            {
                                pHeapAdditionsVector->push_back(pointNeighbors.GetNumAdditions());

                                unsigned int neighborPosition = 0;

                                while (pointNeighbors.HasNext())
                                {
                                    Neighbor neighbor = pointNeighbors.Next();

                                    NeighborExt extNeighbor = {neighbor.pointId, neighbor.distanceSquared, pointId, neighborPosition};

                                    pNeighborsExtVector->push_back(extNeighbor);
                                    ++neighborPosition;
                                }
                            }
                            else
                            {
                                StripePoint stripePoint = {pointId, point.x, point.y, iWindowStripe + windowStartStripe};
                                pPendingPoints->emplace(pointId, stripePoint);
                                pPendingNeighborsContainer->emplace(pointId, std::move(pointNeighbors));
                            }
                        }
                    }
                }
            }

            auto commitFinish = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = commitFinish - commitStart;
            totalElapsedCommit += elapsed;
        }

        void SortNeighbors()
        {
            auto memoryLimit = problemExt.GetMemoryLimitBytes();
            size_t numStripes = pStripeBoundaries->size();
            size_t usedMemory = 4*numStripes*sizeof(size_t)
                                + numStripes*sizeof(StripeBoundaries_t)
                                + 6*64*1024*1024;
            auto safeMemoryLimit = (memoryLimit - usedMemory);

            auto finalSortStart = chrono::high_resolution_clock::now();
            stxxl::sort(pNeighborsExtVector->cbegin(), pNeighborsExtVector->cend(), ExternalNeighborComparer(), safeMemoryLimit);
            auto finalSortEnd = chrono::high_resolution_clock::now();
            elapsedFinalSorting = finalSortEnd - finalSortStart;
        }

        size_t getNumPendingPoints() override
        {
            return pPendingPoints->size();
        }

        size_t getNumStripes() override
        {
            if (pStripeBoundaries != nullptr)
            {
                return pStripeBoundaries->size();
            }
            else
            {
                return 0;
            }
        }

        const chrono::duration<double> getDurationCommitWindow() const override
        {
            return totalElapsedCommit;
        }

        const chrono::duration<double> getDurationFinalSorting() const override
        {
            return elapsedFinalSorting;
        }

        size_t getNumFirstPassWindows() const override
        {
            return numFirstPassWindows;
        }

        size_t getNumSecondPassWindows() const override
        {
            return numSecondPassWindows;
        }

        void SaveToFile() const override
        {
            auto ms = chrono::duration_cast<chrono::milliseconds>(getDuration());

            auto now = chrono::system_clock::now();
            auto in_time_t = chrono::system_clock::to_time_t(now);

            stringstream ss;
            ss << filePrefix << "_" << put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << "_" << ms.count() << ".txt";

            ofstream outFile(ss.str(), ios_base::out);

            size_t numInputPoints = problem.GetInputDatasetSize();
            size_t numNeighbors = problem.GetNumNeighbors();
            size_t pos = 0;

            for (size_t pointId = 1; pointId <=  numInputPoints; ++pointId)
            {
                outFile << pointId;

                for (size_t iNeighbor=0; iNeighbor < numNeighbors; ++iNeighbor)
                {
                    auto& neighbor = pNeighborsExtVector->at(pos);

                    if (neighbor.pointId > 0)
                    {
                        outFile << "\t(" << neighbor.pointId << " " << neighbor.distanceSquared << ")";
                    }
                    else
                    {
                        outFile << "\t(" << "NULL" << " " << neighbor.distanceSquared << ")";
                    }

                    ++pos;
                }

                outFile << endl;
            }

            outFile.close();
        }

        unique_ptr<vector<unsigned long>> FindDifferences(AllKnnResult& result, double accuracy) override
        {
            auto differences = unique_ptr<vector<unsigned long>>(new vector<unsigned long>());

            size_t numInputPoints = problem.GetInputDatasetSize();
            size_t numNeighbors = problem.GetNumNeighbors();
            size_t pos = 0;

            auto& neighborsVector = result.GetNeighborsPriorityQueueVector();

            for (size_t pointId = 1; pointId <=  numInputPoints; ++pointId)
            {
                NeighborsEnumerator* pNeighborsReference = &(neighborsVector.at(pointId - 1));
                vector<Neighbor> removedNeighborsReference;

                for (size_t iNeighbor=0; iNeighbor < numNeighbors; ++iNeighbor)
                {
                    auto& neighbor = pNeighborsExtVector->at(pos);

                    if (pNeighborsReference->HasNext())
                    {
                        Neighbor neighborReference = pNeighborsReference->Next();
                        removedNeighborsReference.push_back(neighborReference);

                        double diff = neighbor.distanceSquared - neighborReference.distanceSquared;

                        if (abs(diff) > accuracy)
                        {
                            differences->push_back(pointId);
                            break;
                        }
                    }
                    else
                    {
                        differences->push_back(pointId);
                        break;
                    }

                    ++pos;
                }

                if (pNeighborsReference->HasNext())
                {
                    differences->push_back(pointId);
                }

                pNeighborsReference->AddAllRemoved(removedNeighborsReference);
            }

            return differences;
        }

        void CalcHeapStats() override
        {
            if (pHeapAdditionsVector == nullptr)
                return;

            minHeapAdditions = numeric_limits<size_t>::max();
            maxHeapAdditions = 0;
            totalHeapAdditions = 0;

            for (auto iter = pHeapAdditionsVector->cbegin(); iter != pHeapAdditionsVector->cend(); ++iter)
            {
                auto additions = *iter;
                if (additions < minHeapAdditions)
                {
                    minHeapAdditions = additions;
                }
                if (additions > maxHeapAdditions)
                {
                    maxHeapAdditions = additions;
                }
                totalHeapAdditions += additions;
            }

            size_t numInputPoints = pHeapAdditionsVector->size();
            avgHeapAdditions = (1.0*totalHeapAdditions)/numInputPoints;
        }

    private:
        bool splitByT = false;
        bool parallelSort = false;
        unique_ptr<pointNeighbors_priority_queue_map_t> pPendingNeighborsContainer;
        unique_ptr<ext_point_vector_t> pStripedInputDataset;
        unique_ptr<ext_point_vector_t> pStripedTrainingDataset;
        unique_ptr<vector<size_t>> pInputStripeOffset;
        unique_ptr<vector<size_t>> pInputStripeCount;
        unique_ptr<vector<size_t>> pTrainingStripeOffset;
        unique_ptr<vector<size_t>> pTrainingStripeCount;
        unique_ptr<vector<StripeBoundaries_t>> pStripeBoundaries;
        bool hasAllocationError = false;
        unique_ptr<unordered_map<unsigned long, StripePoint>> pPendingPoints;
        unique_ptr<ext_neighbors_vector_t> pNeighborsExtVector;
        const AllKnnProblemExternal& problemExt;
        unique_ptr<ext_size_vector_t> pHeapAdditionsVector;
        size_t numFirstPassWindows = 0;
        size_t numSecondPassWindows = 0;
        chrono::duration<double> totalElapsedCommit = chrono::duration<double>(0.0);
        chrono::duration<double> elapsedFinalSorting = chrono::duration<double>(0.0);

        size_t get_optimal_stripes()
        {
            size_t numTrainingPoints = problem.GetTrainingDatasetSize();
            size_t numNeighbors = problem.GetNumNeighbors();

            double numPointsPerDim = sqrt(numTrainingPoints);
            double neighborsPerDim = sqrt(numNeighbors);

            size_t optimal_stripes = llround(numPointsPerDim/neighborsPerDim);
            return optimal_stripes;
        }

        void create_fixed_stripes(size_t numStripes, const ext_point_vector_t& inputDatasetSortedY, const ext_point_vector_t& trainingDatasetSortedY)
        {
            if (splitByT)
                create_fixed_stripes_training(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            else
                create_fixed_stripes_input(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
        }

        void create_fixed_stripes_input(size_t numStripes, const ext_point_vector_t& inputDatasetSortedY, const ext_point_vector_t& trainingDatasetSortedY)
        {
            size_t inputDatasetStripeSize = inputDatasetSortedY.size()/numStripes;
            auto inputDatasetSortedYBegin = inputDatasetSortedY.cbegin();
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();
            auto trainingDatasetSortedYBegin = trainingDatasetSortedY.cbegin();
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();
            auto prevTrainingIterEnd = trainingDatasetSortedYBegin;

            size_t numRemainingPoints = inputDatasetSortedY.size() % numStripes;
            if (numRemainingPoints != 0)
            {
                numStripes += (numRemainingPoints/inputDatasetStripeSize + 1);
            }

            pStripedInputDataset.reset(new ext_point_vector_t());
            pStripedTrainingDataset.reset(new ext_point_vector_t());
            pStripedInputDataset->reserve(inputDatasetSortedY.size());
            pStripedTrainingDataset->reserve(trainingDatasetSortedY.size());

            pInputStripeOffset.reset(new vector<size_t>(numStripes, 0));
            pInputStripeCount.reset(new vector<size_t>(numStripes, 0));
            pTrainingStripeOffset.reset(new vector<size_t>(numStripes, 0));
            pTrainingStripeCount.reset(new vector<size_t>(numStripes, 0));
            pStripeBoundaries.reset(new vector<StripeBoundaries_t>(numStripes, {0.0, 0.0}));

            pPendingPoints.reset(new unordered_map<unsigned long, StripePoint>());
            pPendingNeighborsContainer.reset(new pointNeighbors_priority_queue_map_t());

            for (size_t i=0; i < numStripes; ++i)
            {
                StripeBoundaries_t& stripeBoundaries = pStripeBoundaries->at(i);

                auto inputIterStart = inputDatasetSortedYBegin + i*inputDatasetStripeSize;
                auto inputIterEnd = inputIterStart;
                if ((size_t)distance(inputIterStart, inputDatasetSortedYEnd) <= inputDatasetStripeSize)
                    inputIterEnd = inputDatasetSortedYEnd;
                else
                {
                    inputIterEnd = inputIterStart + inputDatasetStripeSize;
                    auto inputIterEndLimit = inputDatasetSortedYEnd;

                    if ((size_t)distance(inputIterEnd, inputDatasetSortedYEnd) > inputDatasetStripeSize)
                        inputIterEndLimit = inputIterEnd + inputDatasetStripeSize;

                    while (inputIterEnd < inputIterEndLimit && (prev(inputIterEnd))->y == inputIterEnd->y)
                        ++inputIterEnd;
                }

                if (i > 0)
                {
                    while (inputIterStart < inputIterEnd && (prev(inputIterStart))->y == inputIterStart->y)
                        ++inputIterStart;
                }

                pInputStripeOffset->at(i) = (size_t)distance(inputDatasetSortedYBegin, inputIterStart);

                if (inputIterStart < inputIterEnd)
                {
                    point_vector_t inputStripe(inputIterStart, inputIterEnd);
                    pInputStripeCount->at(i) = inputStripe.size();
                    if (parallelSort)
                    {
                        parallel_sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
                             {
                                 return point1.x < point2.x;
                             });

                    }
                    else
                    {
                        std::sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
                             {
                                 return point1.x < point2.x;
                             });
                    }
                    std::copy(inputStripe.begin(), inputStripe.end(), std::back_inserter(*pStripedInputDataset));
                    stripeBoundaries.minY =  i > 0 ? inputIterStart->y : 0.0;
                    stripeBoundaries.maxY =  i < numStripes - 1 ? (inputIterEnd < inputDatasetSortedYEnd ? inputIterEnd->y : 1.0001) : 1.0001;

                    auto trainingIterStart = prevTrainingIterEnd;
                    auto trainingIterEnd = trainingIterStart;

                    while (trainingIterEnd != trainingDatasetSortedYEnd && trainingIterEnd->y < stripeBoundaries.maxY)
                        ++trainingIterEnd;

                    prevTrainingIterEnd = trainingIterEnd;

                    pTrainingStripeOffset->at(i) = (size_t)distance(trainingDatasetSortedYBegin, trainingIterStart);

                    if (trainingIterStart < trainingIterEnd)
                    {
                        point_vector_t trainingStripe(trainingIterStart, trainingIterEnd);
                        pTrainingStripeCount->at(i) = trainingStripe.size();
                        if (parallelSort)
                        {
                            parallel_sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
                             {
                                 return point1.x < point2.x;
                             });

                        }
                        else
                        {
                            std::sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
                             {
                                 return point1.x < point2.x;
                             });
                        }
                        std::copy(trainingStripe.begin(), trainingStripe.end(), std::back_inserter(*pStripedTrainingDataset));
                    }
                }
                else
                {
                    if (inputIterStart >= inputDatasetSortedYEnd)
                    {
                        stripeBoundaries.minY = 1.0001;
                        stripeBoundaries.maxY = 1.0001;
                    }
                    else
                    {
                        stripeBoundaries.minY = inputIterStart->y;
                        stripeBoundaries.maxY = inputIterStart->y;
                    }
                }
            }
        }

        void create_fixed_stripes_training(size_t numStripes, const ext_point_vector_t& inputDatasetSortedY, const ext_point_vector_t& trainingDatasetSortedY)
        {
            size_t trainingDatasetStripeSize = trainingDatasetSortedY.size()/numStripes;
            auto inputDatasetSortedYBegin = inputDatasetSortedY.cbegin();
            auto inputDatasetSortedYEnd = inputDatasetSortedY.cend();
            auto trainingDatasetSortedYBegin = trainingDatasetSortedY.cbegin();
            auto trainingDatasetSortedYEnd = trainingDatasetSortedY.cend();
            auto prevInputIterEnd = inputDatasetSortedYBegin;

            size_t numRemainingPoints = trainingDatasetSortedY.size() % numStripes;
            if (numRemainingPoints != 0)
            {
                numStripes += (numRemainingPoints/trainingDatasetStripeSize + 1);
            }

            pStripedInputDataset.reset(new ext_point_vector_t());
            pStripedTrainingDataset.reset(new ext_point_vector_t());
            pStripedInputDataset->reserve(inputDatasetSortedY.size());
            pStripedTrainingDataset->reserve(trainingDatasetSortedY.size());

            pInputStripeOffset.reset(new vector<size_t>(numStripes, 0));
            pInputStripeCount.reset(new vector<size_t>(numStripes, 0));
            pTrainingStripeOffset.reset(new vector<size_t>(numStripes, 0));
            pTrainingStripeCount.reset(new vector<size_t>(numStripes, 0));
            pStripeBoundaries.reset(new vector<StripeBoundaries_t>(numStripes, {0.0, 0.0}));

            pPendingPoints.reset(new unordered_map<unsigned long, StripePoint>());
            pPendingNeighborsContainer.reset(new pointNeighbors_priority_queue_map_t());

            for (size_t i=0; i < numStripes; ++i)
            {
                StripeBoundaries_t& stripeBoundaries = pStripeBoundaries->at(i);

                auto trainingIterStart = trainingDatasetSortedYBegin + i*trainingDatasetStripeSize;
                auto trainingIterEnd = trainingIterStart;
                if ((size_t)distance(trainingIterStart, trainingDatasetSortedYEnd) <= trainingDatasetStripeSize)
                    trainingIterEnd = trainingDatasetSortedYEnd;
                else
                {
                    trainingIterEnd = trainingIterStart + trainingDatasetStripeSize;
                    auto trainingIterEndLimit = trainingDatasetSortedYEnd;

                    if ((size_t)distance(trainingIterEnd, trainingDatasetSortedYEnd) > trainingDatasetStripeSize)
                        trainingIterEndLimit = trainingIterEnd + trainingDatasetStripeSize;

                    while (trainingIterEnd < trainingIterEndLimit && (prev(trainingIterEnd))->y == trainingIterEnd->y)
                        ++trainingIterEnd;
                }

                if (i > 0)
                {
                    while (trainingIterStart < trainingIterEnd && (prev(trainingIterStart))->y == trainingIterStart->y)
                        ++trainingIterStart;
                }

                pTrainingStripeOffset->at(i) = (size_t)distance(trainingDatasetSortedYBegin, trainingIterStart);

                if (trainingIterStart < trainingIterEnd)
                {
                    point_vector_t trainingStripe(trainingIterStart, trainingIterEnd);
                    pTrainingStripeCount->at(i) = trainingStripe.size();
                    if (parallelSort)
                    {
                            parallel_sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
                                 {
                                     return point1.x < point2.x;
                                 });
                    }
                    else
                    {
                            std::sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
                                 {
                                     return point1.x < point2.x;
                                 });
                    }
                    std::copy(trainingStripe.begin(), trainingStripe.end(), std::back_inserter(*pStripedTrainingDataset));
                    stripeBoundaries.minY =  i > 0 ? trainingIterStart->y : 0.0;
                    stripeBoundaries.maxY =  i < numStripes - 1 ? (trainingIterEnd < trainingDatasetSortedYEnd ? trainingIterEnd->y : 1.0001) : 1.0001;

                    auto inputIterStart = prevInputIterEnd;
                    auto inputIterEnd = inputIterStart;

                    while (inputIterEnd != inputDatasetSortedYEnd && inputIterEnd->y < stripeBoundaries.maxY)
                        ++inputIterEnd;

                    prevInputIterEnd = inputIterEnd;

                    pInputStripeOffset->at(i) = (size_t)distance(inputDatasetSortedYBegin, inputIterStart);

                    if (inputIterStart < inputIterEnd)
                    {
                        point_vector_t inputStripe(inputIterStart, inputIterEnd);
                        pInputStripeCount->at(i) = inputStripe.size();
                        if (parallelSort)
                        {
                            parallel_sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
                                 {
                                     return point1.x < point2.x;
                                 });
                        }
                        else
                        {
                            std::sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
                             {
                                 return point1.x < point2.x;
                             });
                        }
                        std::copy(inputStripe.begin(), inputStripe.end(), std::back_inserter(*pStripedInputDataset));
                    }
                }
                else
                {
                    if (trainingIterStart >= trainingDatasetSortedYEnd)
                    {
                        stripeBoundaries.minY = 1.0001;
                        stripeBoundaries.maxY = 1.0001;
                    }
                    else
                    {
                        stripeBoundaries.minY = trainingIterStart->y;
                        stripeBoundaries.maxY = trainingIterStart->y;
                    }
                }
            }
        }

        bool IsSearchCompleted(const PointNeighbors<neighbors_priority_queue_t>& pointNeighbors)
        {
            size_t lowStripe = pointNeighbors.getLowStripe();
            size_t highStripe = pointNeighbors.getHighStripe();
            size_t numStripes = pStripeBoundaries->size();

            return (lowStripe == 0) && (highStripe >= numStripes-1);
        }
};

#endif // ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
