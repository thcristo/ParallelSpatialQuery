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
#include "DatasetStream.h"

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

        const chrono::duration<double>& getLoadingTime() const { return loadingTime; }

    protected:

    private:
        string inputFilename;
        string trainingFilename;
        size_t numNeighbors = 0;
        unique_ptr<point_vector_t> pInputDataset;
        unique_ptr<point_vector_t> pTrainingDataset;
        chrono::duration<double> loadingTime;

        void LoadDataFiles()
        {
            auto start = chrono::high_resolution_clock::now();

            LoadFile(inputFilename, *pInputDataset);
            LoadFile(trainingFilename, *pTrainingDataset);

            auto finish = chrono::high_resolution_clock::now();
            loadingTime = finish - start;
        }

        void LoadFile(const string& filename, point_vector_t& dataset)
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

        void LoadBinaryFile(const string& filename, point_vector_t& dataset)
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
            //copy(istream_iterator<Point>(fs), istream_iterator<Point>(), back_inserter(dataset));

        }

        void LoadTextFile(const string& filename, point_vector_t& dataset)
        {
            fstream fs(filename, ios::in);
            copy(istream_iterator<Point>(fs), istream_iterator<Point>(), back_inserter(dataset));

            /*
            size_t numInputLines = 0;
            ifstream inputFile(inputFilename, ios_base::in);
            if (inputFile.is_open())
            {
                numInputLines = count(istreambuf_iterator<char>(inputFile), istreambuf_iterator<char>(), '\n');
                if (numInputLines == 0)
                {
                    throw ApplicationException("Input file does not contain any lines.");
                }
            }
            else
            {
                throw ApplicationException("Cannot open input file.");
            }

            size_t numTrainingLines = 0;
            ifstream trainingFile(trainingFilename, ios_base::in);
            if (trainingFile.is_open())
            {
                numTrainingLines = count(istreambuf_iterator<char>(trainingFile), istreambuf_iterator<char>(), '\n');
                if (numTrainingLines == 0)
                {
                    throw ApplicationException("Training file does not contain any lines.");
                }
            }
            else
            {
                throw ApplicationException("Cannot open training file.");
            }

            pInputDataset->reserve(numInputLines);

            pTrainingDataset->reserve(numTrainingLines);

            inputFile.clear();
            inputFile.seekg(0, ios_base::beg);

            trainingFile.clear();
            trainingFile.seekg(0, ios_base::beg);

            LoadPoints(inputFilename, inputFile, *pInputDataset);
            LoadPoints(trainingFilename, trainingFile, *pTrainingDataset);
            */
        }

        /*
        void LoadPoints(const string& filename, ifstream& file, point_vector_t& points)
        {
            long linenum = 0;

            while (file.good())
            {
                string line;
                getline(file, line);
                ++linenum;
                if (line.empty())
                    continue;

                stringstream ss(line);

                Point point = {0, 0.0, 0.0};

                if ( (ss >> point.id) && (ss >> point.x) && (ss >> point.y) )
                {
                    points.push_back(point);
                }
                else
                {
                    stringstream s;
                    s << "Error at reading file " << filename << " at line " << linenum;
                    throw ApplicationException(s.str());
                }
            }
        }
        */

};

#endif // AllKnnPROBLEM_H