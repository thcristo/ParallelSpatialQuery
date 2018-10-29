#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <iomanip>

struct Point {
    long id;
    double x;
    double y;
};

bool endsWith(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

size_t getTargetNumOfPoints(size_t sourcePoints, int factor)
{
    if (factor < 0)
    {
        return sourcePoints/2;
    }
    else if (factor == 1)
    {
        return sourcePoints;
    }
    else if (factor == 4)
    {
        return 4*sourcePoints;
    }
    else if (factor == 2)
    {
        if (sourcePoints % 2 == 0)
        {
            return 2*sourcePoints;
        }
        else
        {
            return 2*(sourcePoints - 1);
        }
    }
    else
        return 0;
}

void write_output_points(const Point& point, std::unique_ptr<std::fstream>& pTargetStream, bool isBinaryTarget, size_t pos, int factor)
{
    if (factor < 0)
    {
        if (isBinaryTarget)
        {
            Point targetPoint = {(long)(pos+1), point.x, point.y};

            pTargetStream->write(reinterpret_cast<const char*>(&targetPoint), std::streamsize(sizeof(Point)));
        }
        else
            *pTargetStream << pos+1 << '\t' << point.x << '\t' << point.y << '\n';
    }
    else if (factor == 1)
    {
        if (isBinaryTarget)
            pTargetStream->write(reinterpret_cast<const char*>(&point), std::streamsize(sizeof(Point)));
        else
            *pTargetStream << point.id << '\t' << point.x << '\t' << point.y << '\n';
    }
    else
    {
        double sourceX = point.x/2.0, sourceY = point.y/2.0;
        Point targetPoints[4];

        targetPoints[0] = {(long)(4*pos+1), sourceX, sourceY};
        targetPoints[1] = {(long)(4*pos+2), sourceX, 0.5 + sourceY};
        targetPoints[2] = {(long)(4*pos+3), 0.5 + sourceX, sourceY};
        targetPoints[3] = {(long)(4*pos+4), 0.5 + sourceX, 0.5 + sourceY};

        for (int i=0; i < 4; ++i)
        {
            if (isBinaryTarget)
                pTargetStream->write(reinterpret_cast<const char*>(targetPoints + i), std::streamsize(sizeof(Point)));
            else
                *pTargetStream << targetPoints[i].id << '\t' << targetPoints[i].x << '\t' << targetPoints[i].y << '\n';
        }
    }
}

std::istream& operator >>(std::istream& i, Point& p)
{
    i >> p.id;
    i >> p.x;
    i >> p.y;

    return i;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cout << "Argument error. Please enter:\n";
        std::cout << "Argument 1: The original filename to use as base\n";
        std::cout << "Argument 2: The target filename to create\n";
        std::cout << "Argument 3: factor (-2, -1, 1, 2 or 4)\n";

        return 1;
    }

    bool isBinaryTarget = false;

    std::string sourceFilename(argv[1]), targetFilename(argv[2]);
    int factor = atoi(argv[3]);

    if (factor != -2 && factor != -1 && factor != 1 && factor != 2 && factor != 4)
    {
        std::cout << "Factor argument must be equal to -2, -1, 1, 2 or 4" << std::endl;
        return 1;
    }

    std::unique_ptr<std::fstream> pTargetStream;

    if (endsWith(targetFilename, ".bin"))
    {
        isBinaryTarget = true;
        pTargetStream.reset(new std::fstream(targetFilename, std::ios::binary | std::ios::out));
    }
    else
    {
        pTargetStream.reset(new std::fstream(targetFilename, std::ios::out));
        if (factor < 0)
            *pTargetStream << std::fixed << std::setprecision(10);
        else
            *pTargetStream << std::fixed << std::setprecision(8);
    }

    if ( !pTargetStream->is_open() )
    {
        std::cout << "Cannot create output file" << std::endl;
        return 1;
    }

    if (endsWith(sourceFilename, ".bin"))
    {
        std::fstream fsSource(sourceFilename, std::ios::in | std::ios::binary);
        if ( !fsSource.is_open() )
        {
            std::cout << "Cannot open input file" << std::endl;
            return 1;
        }
        size_t numPointsSource = 0;
        fsSource.read(reinterpret_cast<char*>(&numPointsSource), std::streamsize(sizeof(size_t)));
        size_t numPointsTarget = getTargetNumOfPoints(numPointsSource, factor);

        if (isBinaryTarget)
        {
            pTargetStream->write(reinterpret_cast<const char*>(&numPointsTarget), std::streamsize(sizeof(size_t)));
        }

        size_t j=0;

        for (size_t i=0; i < numPointsSource && !fsSource.eof(); ++i)
        {
            Point p;
            fsSource.read(reinterpret_cast<char*>(&p), std::streamsize(sizeof(Point)));

            if ((factor == -1 && i%2 == 1) || (factor == -2 && i%2 == 0)
                || factor == 1 || factor == 4 || (factor == 2 && i%2 == 1))
            {
                write_output_points(p, pTargetStream, isBinaryTarget, j, factor);
                ++j;
            }
        }

        fsSource.close();

    }
    else
    {
        std::fstream fsSource(sourceFilename, std::ios::in);
        if ( !fsSource.is_open() )
        {
            std::cout << "Cannot open input file" << std::endl;
            return 1;
        }
        size_t numPointsSource = count(std::istreambuf_iterator<char>(fsSource), std::istreambuf_iterator<char>(), '\n');
        size_t numPointsTarget = getTargetNumOfPoints(numPointsSource, factor);
        if (isBinaryTarget)
        {
            pTargetStream->write(reinterpret_cast<const char*>(&numPointsTarget), std::streamsize(sizeof(size_t)));
        }

        fsSource.clear();
        fsSource.seekg(0, std::ios::beg);

        std::istream_iterator<Point> iter(fsSource);
        std::istream_iterator<Point> eof;

        size_t i =0, j = 0;

        while (iter != eof)
        {
            Point p = *iter++;

            if ((factor == -1 && i%2 == 1) || (factor == -2 && i%2 == 0)
                || factor == 1 || factor == 4 || (factor == 2 && i%2 == 1))
            {
                write_output_points(p, pTargetStream, isBinaryTarget, j, factor);
                ++j;
            }
            ++i;
        }
        fsSource.close();
    }

    pTargetStream->close();

}
