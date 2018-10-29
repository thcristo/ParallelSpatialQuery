/* This file contains the class definition of AkNN problem when external memory is used */

#ifndef ALLKNNPROBLEMEXTERNAL_H
#define ALLKNNPROBLEMEXTERNAL_H

#include "AllKnnProblem.h"


/** \brief Class definition for AkNN problem stored in external memory
 */
class AllKnnProblemExternal : public AllKnnProblem
{
    public:
        AllKnnProblemExternal(const std::string& inputFilename, const std::string& trainingFilename, size_t numNeighbors, bool loadDataFiles, size_t memoryLimitMB)
            : AllKnnProblem(inputFilename, trainingFilename, numNeighbors, false),
                memoryLimitMB(memoryLimitMB), pExtInputDataset(new ext_point_vector_t()), pExtTrainingDataset(new ext_point_vector_t())
        {
            if (loadDataFiles)
                this->LoadExternalDataFiles();
        }

        virtual ~AllKnnProblemExternal() {}

        const ext_point_vector_t& GetExtInputDataset() const
        {
            return *pExtInputDataset;
        }

        const ext_point_vector_t& GetExtTrainingDataset() const
        {
            return *pExtTrainingDataset;
        }

        size_t GetInputDatasetSize() const override
        {
            return pExtInputDataset->size();
        }

        size_t GetTrainingDatasetSize() const override
        {
            return pExtTrainingDataset->size();
        }

        /** \brief Returns the defined memory limit of the external memory algorithm in MB
         *
         * \return size_t memory limit in MB
         *
         */
        size_t GetMemoryLimitBytes() const
        {
            return memoryLimitMB*1024*1024;
        }

    private:
        size_t memoryLimitMB = 0;
        std::unique_ptr<ext_point_vector_t> pExtInputDataset;
        std::unique_ptr<ext_point_vector_t> pExtTrainingDataset;

        void LoadExternalDataFiles()
        {
            auto start = std::chrono::high_resolution_clock::now();

            //call template method of parent class by passing an external memory vector
            LoadFile(inputFilename, *pExtInputDataset);
            LoadFile(trainingFilename, *pExtTrainingDataset);

            auto finish = std::chrono::high_resolution_clock::now();
            loadingTime = finish - start;
        }
};

#endif // ALLKNNPROBLEMEXTERNAL_H
