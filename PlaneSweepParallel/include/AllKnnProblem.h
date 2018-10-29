/* This file contains the class definition for the AkNN problem when internal memory is used */

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

/** \brief Boundaries of a stripe
 */
struct StripeBoundaries_t
{
    double minY;
    double maxY;
};

/** \brief Structure containing stripe data
 */
struct StripeData
{
    const point_vector_vector_t& InputDatasetStripe; /**< vector of input points for each stripe */
    const point_vector_vector_t& TrainingDatasetStripe; /**< vector of training points for each stripe */
    const std::vector<StripeBoundaries_t>& StripeBoundaries; /**< vector of boundaries for each stripe */
};


/** \brief Structure containing stripe data for external memory algorithm
 */
struct StripeDataExternal
{
    const std::vector<size_t>& InputPointsOffset; /**<  vector of offsets for each stripe's input points */
    const std::vector<size_t>& InputPointsCount; /**<  vector of counts for each stripe's input points */
    const std::vector<size_t>& TrainingPointsCount; /**<  vector of counts for each stripe's training points */
    const std::vector<size_t>& TrainingPointsOffset; /**<  vector of offsets for each stripe's training points */
    const std::vector<StripeBoundaries_t>& StripeBoundaries; /**< vector of boundaries for each stripe */
};

/** \brief Overloaded operator used for reading point from a text file
 *
 * \param i istream& input stream
 * \param p Point& point to read
 * \return istream& input stream
 *
 */
std::istream& operator >>(std::istream& i, Point& p)
{
    i >> p.id;
    i >> p.x;
    i >> p.y;

    return i;
}


/** \brief Class definition for AkNN problem stored in internal memory
 */
class AllKnnProblem
{
    public:
        AllKnnProblem(const std::string& inputFilename, const std::string& trainingFilename, size_t numNeighbors, bool loadDataFiles)
            : pInputDataset(new point_vector_t), pTrainingDataset(new point_vector_t)
        {
            //set the filenames and read the data files
            this->inputFilename = inputFilename;
            this->trainingFilename = trainingFilename;
            this->numNeighbors = numNeighbors;
            if (loadDataFiles)
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

        /** \brief Returns the loading time of datasets in ms
         *
         * \return const chrono::duration<double>& loading time in ms
         *
         */
        const std::chrono::duration<double>& getLoadingTime() const { return loadingTime; }

    protected:
        std::string inputFilename;
        std::string trainingFilename;
        std::chrono::duration<double> loadingTime;

        /** \brief Template method for loading data files. It can add the points in an internal or external memory vector
         *
         * \param filename const string& the filename to read from
         * \param dataset PointVector& the internal or external memory vector to add points into
         *
         */
        template<class PointVector>
        void LoadFile(const std::string& filename, PointVector& dataset)
        {
            //Handle both binary and text dataset files
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
        std::unique_ptr<point_vector_t> pInputDataset;
        std::unique_ptr<point_vector_t> pTrainingDataset;

        void LoadDataFiles()
        {
            //Record the time for loading the data files
            auto start = std::chrono::high_resolution_clock::now();

            LoadFile(inputFilename, *pInputDataset);
            LoadFile(trainingFilename, *pTrainingDataset);

            auto finish = std::chrono::high_resolution_clock::now();
            loadingTime = finish - start;
        }

        template<class PointVector>
        void LoadBinaryFile(const std::string& filename, PointVector& dataset)
        {
            //open binary file
            std::fstream fs(filename, std::ios::in | std::ios::binary);
            size_t numPoints = 0;
            //read the number of points at the beginning of the file
            fs.read(reinterpret_cast<char*>(&numPoints), std::streamsize(sizeof(size_t)));
            dataset.reserve(numPoints);

            //read each point and add to vector
            for (size_t i=0; i < numPoints && !fs.eof(); ++i)
            {
                Point p;
                fs.read(reinterpret_cast<char*>(&p), std::streamsize(sizeof(Point)));
                dataset.push_back(p);
            }

            fs.close();
        }

        template<class PointVector>
        void LoadTextFile(const std::string& filename, PointVector& dataset)
        {
            std::fstream fs(filename, std::ios::in);
            std::copy(std::istream_iterator<Point>(fs), std::istream_iterator<Point>(), std::back_inserter(dataset));
            fs.close();
        }
};

#endif // AllKnnPROBLEM_H
