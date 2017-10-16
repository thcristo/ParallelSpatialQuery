#ifndef AllKnnPROBLEM_H
#define AllKnnPROBLEM_H

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "ApplicationException.h"
using namespace std;

typedef struct{
    long id;
    double x;
    double y;
} Point;

typedef struct{
    Point* point;
    double distance;
} Neighbor;

class AllKnnProblem
{
    public:
        AllKnnProblem(string inputFilename, string trainingFilename, int numNeighbors)
        {
            this->inputFilename = inputFilename;
            this->trainingFilename = trainingFilename;
            this->numNeighbors = numNeighbors;
            this->LoadDataFiles();
        }

        virtual ~AllKnnProblem()
        {
            if (pInputDataset)
                delete pInputDataset;
            if (pTrainingDataset)
                delete pTrainingDataset;
        }

        vector<Point>& GetInputDataSet() const
        {
            return *pInputDataset;
        }

        vector<Point>& GetTrainingDataSet() const
        {
            return *pTrainingDataset;
        }

        int GetNumNeighbors() const
        {
            return numNeighbors;
        }

    protected:

    private:
        string inputFilename;
        string trainingFilename;
        int numNeighbors;
        vector<Point>* pInputDataset;
        vector<Point>* pTrainingDataset;

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

            pInputDataset = new vector<Point>();
            pInputDataset->reserve(numInputLines);

            pTrainingDataset = new vector<Point>();
            pTrainingDataset->reserve(numTrainingLines);

            LoadPoints(inputFile, *pInputDataset);
            LoadPoints(trainingFile, *pTrainingDataset);

        }

        void LoadPoints(ifstream& file, vector<Point>& points)
        {
            while (file.good())
            {
                string line;
                getline(file, line);
                if (line.empty())
                    continue;
                stringstream ss(line);

                Point point = {0, 0.0, 0.0};

                ss >> point.id;
                ss >> point.x;
                ss >> point.y;

                points.push_back(point);
            }
        }

};

#endif // AllKnnPROBLEM_H
