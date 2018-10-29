#include <iostream>
#include <string>
#include <fstream>
#include <random>

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
        std::cout << "Argument error. Please enter:\n";
        std::cout << "Argument 1: The number of points to create\n";
        std::cout << "Argument 2: The output filename\n";

        return 1;
    }

    try
    {
        size_t numPoints = strtoul(argv[1], nullptr, 10);
        std::string filename = argv[2];

        std::ofstream outStream(filename, std::ios::binary | std::ios::out);
        if ( !outStream.is_open() )
        {
            return 1;
        }

        outStream.write(reinterpret_cast<const char*>(&numPoints), std::streamsize(sizeof(size_t)));

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0,1.0);

        for (size_t i=0; i < numPoints; ++i)
        {
            Point p = {(long)(i+1), distribution(generator), distribution(generator)};
            outStream.write(reinterpret_cast<const char*>(&p), std::streamsize(sizeof(Point)));
        }

        outStream.close();

        std::cout << numPoints << " points written to file " << filename << std::endl;

        return 0;
    }
    catch(std::exception& ex)
    {
        std::cout << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}
