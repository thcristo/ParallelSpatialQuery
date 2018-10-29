/* This file contains the class definition of an application exception */
#ifndef APPLICATIONEXCEPTION_H
#define APPLICATIONEXCEPTION_H

#include <exception>
#include <string>

/** \brief Application exception class
 */
class ApplicationException : public std::exception
{
    public:
        ApplicationException(const std::string& message) : message(message) {}

        virtual ~ApplicationException() {}

        virtual const char* what() const throw()
        {
            return message.c_str();
        }
    protected:

    private:
        std::string message;
};

#endif // APPLICATIONEXCEPTION_H
