#include <iostream>
#include "NvInferRuntimeCommon.h"


class Logger: public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
}gLogger;


int main () {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}