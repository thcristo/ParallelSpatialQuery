#ifndef AllKnnPROBLEM_H
#define AllKnnPROBLEM_H

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <memory>
#include "ApplicationException.h"
#include "PlaneSweepParallel.h"

using namespace std;


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

        const point_vector_t& GetInputDatasetSorted()
        {
            if (!pInputDatasetSorted)
            {
                pInputDatasetSorted.reset(new point_vector_t(GetInputDataset()));

                sort(pInputDatasetSorted->begin(), pInputDatasetSorted->end(),
                 [](const Point& point1, const Point& point2)
                 {
                     return point1.x < point2.x;
                 });
            }

            return *pInputDatasetSorted;
        }

        const point_vector_t& GetTrainingDatasetSorted()
        {
            if (!pTrainingDatasetSorted)
            {
                pTrainingDatasetSorted.reset(new point_vector_t(GetTrainingDataset()));

                sort(pTrainingDatasetSorted->begin(), pTrainingDatasetSorted->end(),
                 [](const Point& point1, const Point& point2)
                 {
                     return point1.x < point2.x;
                 });
            }

            return *pTrainingDatasetSorted;
        }

        const vector<point_vector_t>& GetTrainingDatasetSortedCopy()
        {
            if (!pTrainingDatasetSortedCopy)
            {
                pTrainingDatasetSortedCopy.reset(new vector<point_vector_t>(8, GetTrainingDatasetSorted()));
            }

            return *pTrainingDatasetSortedCopy;
        }

        void ClearSortedVectors()
        {
            if (pInputDatasetSorted)
            {
                pInputDatasetSorted.reset();
            }

            if (pTrainingDatasetSorted)
            {
                pTrainingDatasetSorted.reset();
            }

            if (pTrainingDatasetSortedCopy)
            {
                pTrainingDatasetSortedCopy.reset();
            }
        }

        size_t GetNumNeighbors() const
        {
            return numNeighbors;
        }

    protected:

    private:
        string inputFilename;
        string trainingFilename;
        size_t numNeighbors;
        unique_ptr<point_vector_t> pInputDataset;
        unique_ptr<point_vector_t> pTrainingDataset;
        unique_ptr<point_vector_t> pInputDatasetSorted;
        unique_ptr<point_vector_t> pTrainingDatasetSorted;

        unique_ptr<vector<point_vector_t>> pTrainingDatasetSortedCopy;

        void LoadDataFiles()
        {
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

        }

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

};

#endif // AllKnnPROBLEM_H
