#ifndef APPLICATIONEXCEPTION_H
#define APPLICATIONEXCEPTION_H

#include <exception>
#include <string>
using namespace std;

class ApplicationException : public exception
{
    public:
        ApplicationException(const string& message) : message(message) {}

        virtual ~ApplicationException() {}

        virtual const char* what() const throw()
        {
            return message.c_str();
        }
    protected:

    private:
        string message;
};

#endif // APPLICATIONEXCEPTION_H
