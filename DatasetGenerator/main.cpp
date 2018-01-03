#include <iostream>
#include <string>
#include <fstream>
#include <random>

using namespace std;

struct Point
{
    long id;
    double x;
    double y;
};

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        cout << "Argument error. Please enter:\n";
        cout << "Argument 1: The number of points to create\n";
        cout << "Argument 2: The output filename\n";

        return 1;
    }

    try
    {
        size_t numPoints = strtoul(argv[1], nullptr, 10);
        string filename = argv[2];

        ofstream outStream(filename, ios::binary | ios::out);
        if ( !outStream.is_open() )
        {
            return 1;
        }

        outStream.write(reinterpret_cast<const char*>(&numPoints), streamsize(sizeof(size_t)));

        default_random_engine generator;
        uniform_real_distribution<double> distribution(0.0,1.0);

        for (size_t i=0; i < numPoints; ++i)
        {
            Point p = {(long)(i+1), distribution(generator), distribution(generator)};
            outStream.write(reinterpret_cast<const char*>(&p), streamsize(sizeof(Point)));
        }

        outStream.close();

        cout << numPoints << " points written to file " << filename << endl;

        return 0;
    }
    catch(exception& ex)
    {
        cout << "Exception: " << ex.what() << endl;
        return 1;
    }
}
