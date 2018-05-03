#ifndef ALLKNNPROBLEMEXTERNAL_H
#define ALLKNNPROBLEMEXTERNAL_H

#include "AllKnnProblem.h"


class AllKnnProblemExternal : public AllKnnProblem
{
    public:
        AllKnnProblemExternal(const string& inputFilename, const string& trainingFilename, size_t numNeighbors, bool loadDataFiles, size_t memoryLimitMB)
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

        size_t GetMemoryLimitBytes() const
        {
            return memoryLimitMB*1024*1024;
        }

    private:
        size_t memoryLimitMB = 0;
        unique_ptr<ext_point_vector_t> pExtInputDataset;
        unique_ptr<ext_point_vector_t> pExtTrainingDataset;

        void LoadExternalDataFiles()
        {
            auto start = chrono::high_resolution_clock::now();

            LoadFile(inputFilename, *pExtInputDataset);
            LoadFile(trainingFilename, *pExtTrainingDataset);

            auto finish = chrono::high_resolution_clock::now();
            loadingTime = finish - start;
        }
};

#endif // ALLKNNPROBLEMEXTERNAL_H
