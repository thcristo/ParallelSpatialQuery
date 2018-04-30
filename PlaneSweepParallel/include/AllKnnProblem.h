#ifndef AllKnnPROBLEM_H
#define AllKnnPROBLEM_H

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>
#include <iterator>
#include "ApplicationException.h"
#include "PlaneSweepParallel.h"

using namespace std;

struct StripeBoundaries_t
{
    double minY;
    double maxY;
};

struct StripeData
{
    const point_vector_vector_t& InputDatasetStripe;
    const point_vector_vector_t& TrainingDatasetStripe;
    const vector<StripeBoundaries_t>& StripeBoundaries;
};

struct StripeDataExternal
{
    const vector<size_t>& InputPointsOffset;
    const vector<size_t>& InputPointsCount;
    const vector<size_t>& TrainingPointsCount;
    const vector<size_t>& TrainingPointsOffset;
    const vector<StripeBoundaries_t>& StripeBoundaries;
};

istream& operator >>(istream& i, Point& p)
{
    i >> p.id;
    i >> p.x;
    i >> p.y;

    return i;
}

class AllKnnProblem
{
    public:
        AllKnnProblem(const string& inputFilename, const string& trainingFilename, size_t numNeighbors)
            : pInputDataset(new point_vector_t), pTrainingDataset(new point_vector_t)
        {
            this->inputFilename = inputFilename;
            this->trainingFilename = trainingFilename;
            this->numNeighbors = numNeighbors;
            this->LoadDataFiles();
        }

        virtual ~AllKnnProblem()
        {
        }

        const point_vector_t& GetInputDataset() const
        {
            return *pInputDataset;
        }

        const point_vector_t& GetTrainingDataset() const
        {
            return *pTrainingDataset;
        }

        size_t GetNumNeighbors() const
        {
            return numNeighbors;
        }

        virtual size_t GetInputDatasetSize() const
        {
            return pInputDataset->size();
        }

        virtual size_t GetTrainingDatasetSize() const
        {
            return pTrainingDataset->size();
        }

        const chrono::duration<double>& getLoadingTime() const { return loadingTime; }

    protected:
        string inputFilename;
        string trainingFilename;
        chrono::duration<double> loadingTime;

        template<class PointVector>
        void LoadFile(const string& filename, PointVector& dataset)
        {
            if (endsWith(filename, ".bin"))
            {
                LoadBinaryFile(filename, dataset);
            }
            else
            {
                LoadTextFile(filename, dataset);
            }
        }

    private:
        size_t numNeighbors = 0;
        unique_ptr<point_vector_t> pInputDataset;
        unique_ptr<point_vector_t> pTrainingDataset;

        void LoadDataFiles()
        {
            auto start = chrono::high_resolution_clock::now();

            LoadFile(inputFilename, *pInputDataset);
            LoadFile(trainingFilename, *pTrainingDataset);

            auto finish = chrono::high_resolution_clock::now();
            loadingTime = finish - start;
        }

        template<class PointVector>
        void LoadBinaryFile(const string& filename, PointVector& dataset)
        {
            fstream fs(filename, ios::in | ios::binary);
            size_t numPoints = 0;
            fs.read(reinterpret_cast<char*>(&numPoints), streamsize(sizeof(size_t)));
            dataset.reserve(numPoints);

            for (size_t i=0; i < numPoints && !fs.eof(); ++i)
            {
                Point p;
                fs.read(reinterpret_cast<char*>(&p), streamsize(sizeof(Point)));
                dataset.push_back(p);
            }
        }

        template<class PointVector>
        void LoadTextFile(const string& filename, PointVector& dataset)
        {
            fstream fs(filename, ios::in);
            copy(istream_iterator<Point>(fs), istream_iterator<Point>(), back_inserter(dataset));
        }
};

#endif // AllKnnPROBLEM_H
