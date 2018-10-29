/* This file contains a class definition of AkNN result for striped plane sweep algorithm when external memory is used */
#ifndef ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
#define ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
#include <stxxl/sort>
#include "AllKnnResultStripes.h"
#include "StripesWindow.h"
#include "AllKnnProblemExternal.h"

/** \brief Comparer for sorting points by y
 */
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

/** \brief Comparer for sorting points by x
 */
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

/** \brief Comparer for sorting neighbors by input point id and neighbor rank
 *          This is used for sorting the result of the algorithm
 */
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

/** \brief Class definition of AkNN result of striped plane sweep algorithm (external memory)
 */
class AllKnnResultStripesParallelExternal : public AllKnnResult
{
    public:
        AllKnnResultStripesParallelExternal(const AllKnnProblemExternal& problem, const std::string& filePrefix)
            : AllKnnResult(problem, filePrefix), problemExt(problem)
        {
        }

        AllKnnResultStripesParallelExternal(const AllKnnProblemExternal& problem, const std::string& filePrefix, bool parallelSort, bool splitByT)
            : AllKnnResult(problem, filePrefix), splitByT(splitByT), parallelSort(parallelSort), problemExt(problem)
        {
        }

        virtual ~AllKnnResultStripesParallelExternal()
        {
        }

        /** \brief Splits the datasets into stripes
         *
         * \param numStripes size_t the desired number of stripes
         * \return size_t the actual number of stripes
         *
         */
        size_t SplitStripes(size_t numStripes)
        {
            //copy both datasets
            ext_point_vector_t inputDatasetSortedY(problemExt.GetExtInputDataset());
            ext_point_vector_t trainingDatasetSortedY(problemExt.GetExtTrainingDataset());

            //calculate memory limit based on variables
            size_t usedMemory = 4*64*1024*1024;
            auto memoryLimit = problemExt.GetMemoryLimitBytes();
            auto safeMemoryLimit = memoryLimit - usedMemory;

            //call external memory sort routine of STXXL library
            stxxl::sort(inputDatasetSortedY.cbegin(), inputDatasetSortedY.cend(), ExternalPointComparerY(), safeMemoryLimit);
            stxxl::sort(trainingDatasetSortedY.cbegin(), trainingDatasetSortedY.cend(), ExternalPointComparerY(), safeMemoryLimit);

            //check if specific number of stripes has been requested
            if (numStripes > 0)
            {
                //split datasets into this number of stripes
                create_fixed_stripes(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            }
            else
            {
                //find the optimal number of stripes
                numStripes = get_optimal_stripes();
                create_fixed_stripes(numStripes, inputDatasetSortedY, trainingDatasetSortedY);
            }

            return pStripeBoundaries->size();
        }

        /** \brief Returns a set of stripes that fits into RAM
         *
         * \param fromStripe size_t index of starting stripe
         * \param secondPass bool true if this is the second phase of the algorithm (higher to lower y)
         * \return unique_ptr<StripesWindow> the window of stripes
         *
         */
        std::unique_ptr<StripesWindow> GetWindow(size_t fromStripe, bool secondPass)
        {
            auto memoryLimit = problemExt.GetMemoryLimitBytes();
            auto safeMemoryLimit = 9*memoryLimit/10;

            size_t numNeighbors = problem.GetNumNeighbors();
            size_t numStripes = pStripeBoundaries->size();
            //estimation of required memory based on variables, this is an approximation with some safety factors
            size_t usedMemory = (pPendingPoints->size()*(sizeof(unsigned long) + sizeof(StripePoint)))
                                + (pPendingNeighborsContainer->size()*(sizeof(unsigned long) + sizeof(PointNeighbors<neighbors_priority_queue_t>) + numNeighbors*sizeof(Neighbor)))
                                + (pPendingPoints->size()*(sizeof(Point)))
                                + 2*(pPendingPoints->size()*(2*sizeof(void*)))
                                + 4*numStripes*sizeof(size_t)
                                + numStripes*sizeof(StripeBoundaries_t)
                                + 6*64*1024*1024;

            std::cout << "pending points " << pPendingPoints->size() << " " << pPendingNeighborsContainer->size() << std::endl;
            std::cout << "reserved memory " << usedMemory << std::endl;

            if (secondPass)
            {
                //in the second phase we return training points only
                size_t endStripe = fromStripe;
                size_t startStripe = endStripe + 1;

                //find how many stripes can fit into available memory
                do
                {
                    //add stripes of training points until we reach the memory limit
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
                    //we cannot allocate even a single stripe so the algorithm reports memory allocation error
                    //this can happen if we have too many pending points
                    hasAllocationError = true;
                    return std::unique_ptr<StripesWindow>(nullptr);
                }

                size_t numWindowStripes = endStripe - startStripe + 1;
                std::unique_ptr<point_vector_vector_t> pTrainingStripes(new point_vector_vector_t(numWindowStripes));
                std::unique_ptr<std::vector<StripeBoundaries_t>> pBoundaries(new std::vector<StripeBoundaries_t>(numWindowStripes));

                //add stripes to current window until we reach the memory limit
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

                    //store the boundaries of each stripe
                    boundaries.minY = pStripeBoundaries->at(iStripe).minY;
                    boundaries.maxY = pStripeBoundaries->at(iStripe).maxY;
                }

                std::unique_ptr<StripesWindow> pWindow(new StripesWindow(startStripe, endStripe, pTrainingStripes, pBoundaries));
                ++numSecondPassWindows;
                return pWindow;
            }
            else
            {
                //this is the first phase of the algorithm, we return input and training point stripes
                size_t startStripe = fromStripe;
                size_t endStripe = startStripe;

                //find how many stripes can fit into available memory
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
                    //we cannot allocate even a single stripe so the algorithm reports memory allocation error
                    //this can happen if we have too many pending points
                    hasAllocationError = true;
                    return std::unique_ptr<StripesWindow>(nullptr);
                }

                size_t numWindowStripes = endStripe - startStripe;

                std::unique_ptr<point_vector_vector_t> pInputStripes(new point_vector_vector_t(numWindowStripes)), pTrainingStripes(new point_vector_vector_t(numWindowStripes));
                std::unique_ptr<std::vector<StripeBoundaries_t>> pBoundaries(new std::vector<StripeBoundaries_t>(numWindowStripes));

                //add stripes to current window until we reach the memory limit
                for (size_t iStripe = startStripe; iStripe < endStripe; ++iStripe)
                {
                    auto& inputPoints = pInputStripes->at(iStripe-startStripe);
                    auto& trainingPoints = pTrainingStripes->at(iStripe-startStripe);
                    auto& boundaries = pBoundaries->at(iStripe-startStripe);

                    //add input point stripes
                    if (pInputStripeCount->at(iStripe) > 0)
                    {
                        auto inputStart = pStripedInputDataset->cbegin() + pInputStripeOffset->at(iStripe);
                        auto inputEnd = inputStart + pInputStripeCount->at(iStripe);
                        inputPoints.reserve(pInputStripeCount->at(iStripe));
                        std::copy(inputStart, inputEnd, std::back_inserter(inputPoints));
                    }

                    //add training point stripes
                    if (pTrainingStripeCount->at(iStripe) > 0)
                    {
                        auto trainingStart = pStripedTrainingDataset->cbegin() + pTrainingStripeOffset->at(iStripe);
                        auto trainingEnd = trainingStart + pTrainingStripeCount->at(iStripe);
                        trainingPoints.reserve(pTrainingStripeCount->at(iStripe));
                        std::copy(trainingStart, trainingEnd, std::back_inserter(trainingPoints));
                    }

                    //store the boundaries of each stripe
                    boundaries.minY = pStripeBoundaries->at(iStripe).minY;
                    boundaries.maxY = pStripeBoundaries->at(iStripe).maxY;
                }

                std::unique_ptr<StripesWindow> pWindow(new StripesWindow(startStripe, endStripe-1, pInputStripes, pTrainingStripes, pBoundaries, numNeighbors));
                ++numFirstPassWindows;
                return pWindow;
            }
        }

        /** \brief Returns true if an allocation error happened
         *
         * \return bool
         *
         */
        bool HasAllocationError() override
        {
            return hasAllocationError;
        }

        /** \brief Returns a list of pending points to be examined for a specified window
         *
         * \param window const StripesWindow& the window of stripes that is currently in process
         * \return unique_ptr<point_vector_t> the vector of pending points
         *
         */
        std::unique_ptr<point_vector_t> GetPendingPointsForWindow(const StripesWindow& window)
        {
            std::unique_ptr<point_vector_t> pPoints(new point_vector_t());
            pPoints->reserve(pPendingPoints->size());
            auto pendingPointsBegin = pPendingPoints->cbegin();
            auto pendingPointsEnd = pPendingPoints->cend();

            if (window.IsSecondPass())
            {
                size_t startStripe = window.GetStartStripe();

                //for second phase we want to examine all pending points except those
                // with neighbors having a low stripe lower than the first stripe of current window
                for (auto pendingIter = pendingPointsBegin; pendingIter != pendingPointsEnd; ++pendingIter)
                {
                    auto pointId = pendingIter->second.id;
                    //find the neighbors of pending points by looking at a hash table
                    auto& neighbors = pPendingNeighborsContainer->at(pointId);
                    //get the lowest stripe examined so far
                    size_t lowStripe = neighbors.getLowStripe();

                    //if lowest stripe examined is lower than the first stripe of current window,
                    //we do not need to examine this pending point
                    if (lowStripe > startStripe)
                    {
                        pPoints->push_back({pendingIter->second.id, pendingIter->second.x, pendingIter->second.y});
                    }
                }
            }
            else
            {
                size_t endStripe = window.GetEndStripe();

                //for first phase we want all pending points
                for (auto pendingIter = pendingPointsBegin; pendingIter != pendingPointsEnd; ++pendingIter)
                {
                    auto pointId = pendingIter->second.id;
                    //find the neighbors of pending points by looking at a hash table
                    auto& neighbors = pPendingNeighborsContainer->at(pointId);
                    //get the highest stripe examined so far
                    size_t highStripe = neighbors.getHighStripe();

                    //if highest stripe examined is higher than the last stripe of current window,
                    //we do not need to examine this pending point
                    if (highStripe < endStripe)
                    {
                        pPoints->push_back({pointId, pendingIter->second.x, pendingIter->second.y});
                    }
                }
            }

            return pPoints;
        }

        /** \brief Returns the hash table of neighbors fo pending points
         *
         * \return pointNeighbors_priority_queue_map_t&
         *
         */
        pointNeighbors_priority_queue_map_t& GetPendingNeighborsContainer()
        {
            return *pPendingNeighborsContainer;
        }

        /** \brief Transfers all the completed points to STXXL vectors. It checks points in the given window and pending points
         *
         * \param window StripesWindow& the window of stripes to examine
         * \param pendingPoints point_vector_t& the list of pending points to examine
         * \return void
         *
         */
        void CommitWindow(StripesWindow& window, point_vector_t& pendingPoints)
        {
            auto commitStart = std::chrono::high_resolution_clock::now();

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

            //check pending list for any completed points
            for (auto pendingPointsIter = pendingPointsBegin; pendingPointsIter < pendingPointsEnd; ++pendingPointsIter)
            {
                auto pointId = pendingPointsIter->id;
                auto& pointNeighbors = pPendingNeighborsContainer->at(pointId);
                //check if search has been completed
                if (IsSearchCompleted(pointNeighbors))
                {
                    //transfer heap statistics
                    pHeapAdditionsVector->push_back(pointNeighbors.GetNumAdditions());

                    unsigned int neighborPosition = 0;

                    while (pointNeighbors.HasNext())
                    {
                        Neighbor neighbor = pointNeighbors.Next();

                        //we need to store the input point id and rank so we can later sort the neighbors
                        NeighborExt extNeighbor = {neighbor.pointId, neighbor.distanceSquared, pointId, neighborPosition};

                        //store the neighbors in external memory vector
                        pNeighborsExtVector->push_back(extNeighbor);
                        ++neighborPosition;
                    }
                    //remove pending points and neighbors from RAM
                    pPendingPoints->erase(pointId);
                    pPendingNeighborsContainer->erase(pointId);
                }
            }

            bool isSecondPass = window.IsSecondPass();
            if (!isSecondPass)
            {
                //check window of stripes for completed points
                auto stripeData = window.GetStripeData();
                auto& neighborsContainer = window.GetNeighborsContainer();
                size_t numWindowStripes = window.GetNumStripes();
                size_t windowStartStripe = window.GetStartStripe();

                //examine all stripes of the window
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
                                //transfer heap statistics
                                pHeapAdditionsVector->push_back(pointNeighbors.GetNumAdditions());

                                unsigned int neighborPosition = 0;

                                while (pointNeighbors.HasNext())
                                {
                                    Neighbor neighbor = pointNeighbors.Next();

                                    //we need to store the input point id and rank so we can later sort the neighbors
                                    NeighborExt extNeighbor = {neighbor.pointId, neighbor.distanceSquared, pointId, neighborPosition};

                                    //store the neighbors in external memory vector
                                    pNeighborsExtVector->push_back(extNeighbor);
                                    ++neighborPosition;
                                }
                            }
                            else
                            {
                                //if search of neighbors has not been completed, move the point and its so far found neighbors in the pending points list
                                StripePoint stripePoint = {pointId, point.x, point.y, iWindowStripe + windowStartStripe};
                                pPendingPoints->emplace(pointId, stripePoint);
                                pPendingNeighborsContainer->emplace(pointId, std::move(pointNeighbors));
                            }
                        }
                    }
                }
            }

            //we record the time for reporting purposes
            auto commitFinish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = commitFinish - commitStart;
            totalElapsedCommit += elapsed;
        }

        /** \brief Sorts the final result by input point id and neighbor rank
         *
         * \return void
         *
         */
        void SortNeighbors()
        {
            if (hasAllocationError)
                return;

            auto memoryLimit = problemExt.GetMemoryLimitBytes();
            size_t numStripes = pStripeBoundaries->size();
            size_t usedMemory = 4*numStripes*sizeof(size_t)
                                + numStripes*sizeof(StripeBoundaries_t)
                                + 6*64*1024*1024;

            auto safeMemoryLimit = (memoryLimit - usedMemory);

            auto finalSortStart = std::chrono::high_resolution_clock::now();
            //call the STXXL sort routine to sort neighbors
            stxxl::sort(pNeighborsExtVector->cbegin(), pNeighborsExtVector->cend(), ExternalNeighborComparer(), safeMemoryLimit);
            auto finalSortEnd = std::chrono::high_resolution_clock::now();
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

        const std::chrono::duration<double> getDurationCommitWindow() const override
        {
            return totalElapsedCommit;
        }

        const std::chrono::duration<double> getDurationFinalSorting() const override
        {
            return elapsedFinalSorting;
        }

        /** \brief Returns the number of windows processed in the first phase
         *
         * \return size_t
         *
         */
        size_t getNumFirstPassWindows() const override
        {
            return numFirstPassWindows;
        }

        /** \brief Returns the number of windows processed in the second phase
         *
         * \return size_t
         *
         */
        size_t getNumSecondPassWindows() const override
        {
            return numSecondPassWindows;
        }

        /** \brief Saves neighbors found for each input point to a text file
         *
         */
        void SaveToFile() const override
        {
            if (hasAllocationError)
                return;

            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(getDuration());

            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);

            std::stringstream ss;
            ss << filePrefix << "_" << std::put_time(localtime(&in_time_t), "%Y%m%d%H%M%S") << "_" << ms.count() << ".txt";

            std::ofstream outFile(ss.str(), std::ios_base::out);

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

                outFile << std::endl;
            }

            outFile.close();
        }

        /** \brief Compares a result with a reference result to find any differences in distances of neighbors
         *
         * \param result AllKnnResult& the result to check for differences
         * \param accuracy double the accuracy to use for comparisons
         * \return unique_ptr<vector<unsigned long>>  vector of input point ids where differences exist
         *
         */
        std::unique_ptr<std::vector<unsigned long>> FindDifferences(AllKnnResult& result, double accuracy) override
        {
            if (hasAllocationError)
                return std::unique_ptr<std::vector<unsigned long>>(nullptr);

            auto differences = std::unique_ptr<std::vector<unsigned long>>(new std::vector<unsigned long>());

            size_t numInputPoints = problem.GetInputDatasetSize();
            size_t numNeighbors = problem.GetNumNeighbors();
            size_t pos = 0;

            auto& neighborsVector = result.GetNeighborsPriorityQueueVector();

            for (size_t pointId = 1; pointId <=  numInputPoints; ++pointId)
            {
                NeighborsEnumerator* pNeighborsReference = &(neighborsVector.at(pointId - 1));
                std::vector<Neighbor> removedNeighborsReference;

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

         /** \brief Calculates heap statistics for reporting purposes
         */
        void CalcHeapStats() override
        {
            if (hasAllocationError)
                return;

            if (pHeapAdditionsVector == nullptr)
                return;

            minHeapAdditions = std::numeric_limits<size_t>::max();
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
        std::unique_ptr<pointNeighbors_priority_queue_map_t> pPendingNeighborsContainer;
        std::unique_ptr<ext_point_vector_t> pStripedInputDataset;
        std::unique_ptr<ext_point_vector_t> pStripedTrainingDataset;
        std::unique_ptr<std::vector<size_t>> pInputStripeOffset;
        std::unique_ptr<std::vector<size_t>> pInputStripeCount;
        std::unique_ptr<std::vector<size_t>> pTrainingStripeOffset;
        std::unique_ptr<std::vector<size_t>> pTrainingStripeCount;
        std::unique_ptr<std::vector<StripeBoundaries_t>> pStripeBoundaries;
        bool hasAllocationError = false;
        std::unique_ptr<std::unordered_map<unsigned long, StripePoint>> pPendingPoints;
        std::unique_ptr<ext_neighbors_vector_t> pNeighborsExtVector;
        const AllKnnProblemExternal& problemExt;
        std::unique_ptr<ext_size_vector_t> pHeapAdditionsVector;
        size_t numFirstPassWindows = 0;
        size_t numSecondPassWindows = 0;
        std::chrono::duration<double> totalElapsedCommit = std::chrono::duration<double>(0.0);
        std::chrono::duration<double> elapsedFinalSorting = std::chrono::duration<double>(0.0);

        /** \brief calculate an optimal number of stripes based on the number of training points and neighbors
         *
         * \return size_t the optimal number of stripes
         *
         */
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

        /** \brief Splits the datasets into stripes based on the input dataset (fixed number of input points per stripe)
         *
         * \param numStripes size_t the desired number of stripes
         * \param inputDatasetSortedY const ext_point_vector_t& the sorted input dataset
         * \param trainingDatasetSortedY const ext_point_vector_t& the sorted training dataset
         * \return void
         *
         */
        void create_fixed_stripes_input(size_t numStripes, const ext_point_vector_t& inputDatasetSortedY, const ext_point_vector_t& trainingDatasetSortedY)
        {
            //The implementation is similar to AllKnnResultStripesParallel.create_fixed_stripes_input
            //with the difference of using external memory vectors and a serial for loop instead of parallel

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

            pInputStripeOffset.reset(new std::vector<size_t>(numStripes, 0));
            pInputStripeCount.reset(new std::vector<size_t>(numStripes, 0));
            pTrainingStripeOffset.reset(new std::vector<size_t>(numStripes, 0));
            pTrainingStripeCount.reset(new std::vector<size_t>(numStripes, 0));
            pStripeBoundaries.reset(new std::vector<StripeBoundaries_t>(numStripes, {0.0, 0.0}));

            pPendingPoints.reset(new std::unordered_map<unsigned long, StripePoint>());
            pPendingNeighborsContainer.reset(new pointNeighbors_priority_queue_map_t());

            //we cannot use a parallel loop because the external memory vectors do not allow concurrent access by multiple threads
            for (size_t i=0; i < numStripes; ++i)
            {
                StripeBoundaries_t& stripeBoundaries = pStripeBoundaries->at(i);

                auto inputIterStart = inputDatasetSortedYBegin + i*inputDatasetStripeSize;
                auto inputIterEnd = inputIterStart;
                if ((size_t)std::distance(inputIterStart, inputDatasetSortedYEnd) <= inputDatasetStripeSize)
                    inputIterEnd = inputDatasetSortedYEnd;
                else
                {
                    inputIterEnd = inputIterStart + inputDatasetStripeSize;
                    auto inputIterEndLimit = inputDatasetSortedYEnd;

                    if ((size_t)std::distance(inputIterEnd, inputDatasetSortedYEnd) > inputDatasetStripeSize)
                        inputIterEndLimit = inputIterEnd + inputDatasetStripeSize;

                    while (inputIterEnd < inputIterEndLimit && (std::prev(inputIterEnd))->y == inputIterEnd->y)
                        ++inputIterEnd;
                }

                if (i > 0)
                {
                    while (inputIterStart < inputIterEnd && (std::prev(inputIterStart))->y == inputIterStart->y)
                        ++inputIterStart;
                }

                pInputStripeOffset->at(i) = (size_t)std::distance(inputDatasetSortedYBegin, inputIterStart);

                if (inputIterStart < inputIterEnd)
                {
                    point_vector_t inputStripe(inputIterStart, inputIterEnd);
                    pInputStripeCount->at(i) = inputStripe.size();
                    if (parallelSort)
                    {
                        tbb::parallel_sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
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

                    pTrainingStripeOffset->at(i) = (size_t)std::distance(trainingDatasetSortedYBegin, trainingIterStart);

                    if (trainingIterStart < trainingIterEnd)
                    {
                        point_vector_t trainingStripe(trainingIterStart, trainingIterEnd);
                        pTrainingStripeCount->at(i) = trainingStripe.size();
                        if (parallelSort)
                        {
                            tbb::parallel_sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
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

        /** \brief Splits the datasets into stripes based on the training dataset (fixed number of training points per stripe)
         *
         * \param numStripes size_t the desired number of stripes
         * \param inputDatasetSortedY const ext_point_vector_t& the sorted input dataset
         * \param trainingDatasetSortedY const ext_point_vector_t& the sorted training dataset
         * \return void
         *
         */
        void create_fixed_stripes_training(size_t numStripes, const ext_point_vector_t& inputDatasetSortedY, const ext_point_vector_t& trainingDatasetSortedY)
        {
            //The implementation is similar to AllKnnResultStripesParallel.create_fixed_stripes_training
            //with the difference of using external memory vectors and a serial for loop instead of parallel

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

            pInputStripeOffset.reset(new std::vector<size_t>(numStripes, 0));
            pInputStripeCount.reset(new std::vector<size_t>(numStripes, 0));
            pTrainingStripeOffset.reset(new std::vector<size_t>(numStripes, 0));
            pTrainingStripeCount.reset(new std::vector<size_t>(numStripes, 0));
            pStripeBoundaries.reset(new std::vector<StripeBoundaries_t>(numStripes, {0.0, 0.0}));

            pPendingPoints.reset(new std::unordered_map<unsigned long, StripePoint>());
            pPendingNeighborsContainer.reset(new pointNeighbors_priority_queue_map_t());

            //we cannot use a parallel loop because the external memory vectors do not allow concurrent access by multiple threads
            for (size_t i=0; i < numStripes; ++i)
            {
                StripeBoundaries_t& stripeBoundaries = pStripeBoundaries->at(i);

                auto trainingIterStart = trainingDatasetSortedYBegin + i*trainingDatasetStripeSize;
                auto trainingIterEnd = trainingIterStart;
                if ((size_t)std::distance(trainingIterStart, trainingDatasetSortedYEnd) <= trainingDatasetStripeSize)
                    trainingIterEnd = trainingDatasetSortedYEnd;
                else
                {
                    trainingIterEnd = trainingIterStart + trainingDatasetStripeSize;
                    auto trainingIterEndLimit = trainingDatasetSortedYEnd;

                    if ((size_t)std::distance(trainingIterEnd, trainingDatasetSortedYEnd) > trainingDatasetStripeSize)
                        trainingIterEndLimit = trainingIterEnd + trainingDatasetStripeSize;

                    while (trainingIterEnd < trainingIterEndLimit && (std::prev(trainingIterEnd))->y == trainingIterEnd->y)
                        ++trainingIterEnd;
                }

                if (i > 0)
                {
                    while (trainingIterStart < trainingIterEnd && (std::prev(trainingIterStart))->y == trainingIterStart->y)
                        ++trainingIterStart;
                }

                pTrainingStripeOffset->at(i) = (size_t)std::distance(trainingDatasetSortedYBegin, trainingIterStart);

                if (trainingIterStart < trainingIterEnd)
                {
                    point_vector_t trainingStripe(trainingIterStart, trainingIterEnd);
                    pTrainingStripeCount->at(i) = trainingStripe.size();
                    if (parallelSort)
                    {
                            tbb::parallel_sort(trainingStripe.begin(), trainingStripe.end(), [](const Point& point1, const Point& point2)
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

                    pInputStripeOffset->at(i) = (size_t)std::distance(inputDatasetSortedYBegin, inputIterStart);

                    if (inputIterStart < inputIterEnd)
                    {
                        point_vector_t inputStripe(inputIterStart, inputIterEnd);
                        pInputStripeCount->at(i) = inputStripe.size();
                        if (parallelSort)
                        {
                            tbb::parallel_sort(inputStripe.begin(), inputStripe.end(), [](const Point& point1, const Point& point2)
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

        /** \brief Returns true if search of k nearest neighbors neighbors has been completed
         *
         * \param pointNeighbors const PointNeighbors<neighbors_priority_queue_t>& the list of k neighbors of a specific input point
         * \return bool
         *
         */
        bool IsSearchCompleted(const PointNeighbors<neighbors_priority_queue_t>& pointNeighbors)
        {
            size_t lowStripe = pointNeighbors.getLowStripe();
            size_t highStripe = pointNeighbors.getHighStripe();
            size_t numStripes = pStripeBoundaries->size();

            return (lowStripe == 0) && (highStripe >= numStripes-1);
        }
};

#endif // ALLKNNRESULTSTRIPESPARALLELEXTERNAL_H
