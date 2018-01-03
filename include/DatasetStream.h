#ifndef DATASETSTREAM_H
#define DATASETSTREAM_H

#include <fstream>
#include "PlaneSweepParallel.h"

class DatasetStream : public ifstream
{
    public:
        DatasetStream(const string& filename, ios_base::openmode mode = ios_base::in) : ifstream(filename, mode),
            filename(filename), mode(mode)
        {
        }
        virtual ~DatasetStream() {}

        const string& getFilename()
        {
            return filename;
        }

        ios_base::openmode getMode()
        {
            return mode;
        }

    protected:

    private:
        string filename;
        ios_base::openmode mode;
};
/*
istream& operator >>(istream& i, Point& p)
{
    DatasetStream* pDS = dynamic_cast<DatasetStream*>(&i);
    if (pDS != nullptr)
    {
        if (pDS->getMode() & ios::binary)
        {
            i.read(reinterpret_cast<char*>(&p), streamsize(sizeof(Point)));
        }
        else
        {
            i >> p.id;
            i >> p.x;
            i >> p.y;
        }
    }
    else
    {
        i >> p.id;
        i >> p.x;
        i >> p.y;
    }

    return i;
}
*/

#endif // DATASETSTREAM_H
