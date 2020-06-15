#include "opi_logger.h"

namespace OPI
{

int Logger::verboseLevel = 0;
std::string Logger::prefix = "[OPI] ";
std::ofstream Logger::logFile;
std::stringstream Logger::messageBuffer;
std::ostream* Logger::redirect = nullptr;
Logger::mode Logger::currentMode = LOGMODE_STDOUT;

Logger::Logger()
{

}

Logger::~Logger()
{
    if (logFile.is_open()) {
        logFile.close();
    }
}

void Logger::setVerboseLevel(int level)
{
	verboseLevel = level;
}

void Logger::setPrefix(std::string newPrefix)
{
    if (newPrefix == "")
    {
        prefix = newPrefix;
    }
    else {
        prefix = "[" + newPrefix + "] ";
    }
}

void Logger::setMode(mode newMode, std::string fileName, std::ostream* redirectTo, bool append)
{
    currentMode = newMode;
    if (logFile.is_open()) {
        logFile.flush();
        logFile.close();
    }
    if (((newMode == LOGMODE_FILE) || (newMode == LOGMODE_BINARY)) && fileName != "") {
        if (append)
        {
            logFile.open(fileName, std::ofstream::out | std::ofstream::app);
        }
        else {
            logFile.open(fileName, std::ofstream::out);
        }
    }
    else if (newMode == LOGMODE_REDIRECT)
    {
        redirect = redirectTo;
    }
}

int Logger::getVerboseLevel()
{
	return verboseLevel;
}

std::string Logger::getBuffer()
{
    std::string contents = messageBuffer.str();
    messageBuffer.str(std::string());
    messageBuffer.clear();
    return contents;
}

std::ostream& Logger::out(int minVerboseLevel)
{
    if (minVerboseLevel > verboseLevel)
    {
        static nullstream dummy;
        return dummy;
    }
    else if (currentMode == LOGMODE_FILE && logFile.is_open())
    {
        return logFile << prefix;
    }
    else if (currentMode == LOGMODE_BINARY && logFile.is_open())
    {
        return logFile;
    }
    else if (currentMode == LOGMODE_BUFFER) {
        return messageBuffer << prefix;
    }
    else if (currentMode == LOGMODE_REDIRECT) {
        return *redirect << prefix;
    }
    else {
        return std::cout << prefix;
    }
}

std::ostream* Logger::getStream()
{
    if (currentMode == LOGMODE_FILE && logFile.is_open())
    {
        return static_cast<std::ostream*>(&logFile);
    }
    else if (currentMode == LOGMODE_BINARY && logFile.is_open())
    {
        return static_cast<std::ostream*>(&logFile);
    }
    else if (currentMode == LOGMODE_BUFFER) {
        return static_cast<std::ostream*>(&messageBuffer);
    }
    else {
        return &std::cout;
    }
}

}
