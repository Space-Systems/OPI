#ifndef OPI_LOGGER_CPP_H
#define OPI_LOGGER_CPP_H

#include "opi_common.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace OPI
{

/// \brief Simple message and debug output handler.
///
/// The OPI Logger is a simple message output handler. Its purpose is to
/// replace std::cout for generating console output, and it can be used in exactly
/// that fashion, with some added benefits. Users can specify
/// a debug level for each message which will only be printed if the global verbose
/// level is set to at least that number. The whole output can be redirected to a
/// log file or buffer at any time. Most of these functions are static so calls to
/// setVerboseLevel() and setMode() affect all output inside OPI.
class Logger
{
	public:

        /// \brief Enum to set the output mode (see setMode() for details).
        OPI_API_EXPORT enum mode {
            LOGMODE_STDOUT,
            LOGMODE_FILE,
            LOGMODE_BINARY,
            LOGMODE_BUFFER,
            LOGMODE_REDIRECT
        };

        /// \brief Class constructor.
        OPI_API_EXPORT Logger();

        /// \brief Class destructor.
        OPI_API_EXPORT ~Logger();

        /// \brief Use the Logger by sending your messages to this function.
        ///
        /// Logger::out() can be used exactly like std::cout in your program: Instead
        /// of \code std::cout << "Text" << endl; \endcode you would write
        /// \code Logger::out() << "Text" << endl; \endcode The parameter minVerboseLevel
        /// contains the verbose level that needs to be set for this message to be shown. For
        /// example, a message sent with
        /// \code Logger::out(3) << "Text" << endl; \endcode
        /// will only be displayed if
        /// \code setVerboseLevel(3); \endcode was previously called anywhere
        /// in the program. The initial value is zero for both, meaning that all messages are
        /// directed to STDOUT by default. See the description of setVerboseLevel() for a
        /// list of what the different verbose levels mean.
        /// \param minVerboseLevel Set the verbose level for this message.
        OPI_API_EXPORT static std::ostream& out(int minVerboseLevel = 0);

        /// \brief Use this function to control where messages go.
        ///
        /// Valid modes are STDOUT, FILE, BINARY, BUFFER and REDIRECT. STDOUT simply writes messages
        /// to the console. FILE requires a file names as an additional parameter.
        /// Setting this mode will cause all output to be redirected to this file.
        /// BINARY is the same, with the exception that any prefixes are ignored.
        /// The file will be closed when changing mode again or when the program
        /// terminates. BUFFER will write all output to a message buffer which can be
        /// retrieved with the getBuffer() function; this can be used to redirect output
        /// to a GUI or any other custom message handler.
        /// With REDIRECT, a pointer to another ostream can be given that all output
        /// will be sent to.
        OPI_API_EXPORT static void setMode(mode newMode, const char* fileName = "", std::ostream* redirectTo = nullptr, bool append = true);

        /// \brief Use this function to control how many messages are shown.
        ///
        /// Each individual message sent via Logger::out() contains a number stating
        /// the minimum verbose level that needs to be set for that message to be
        /// displayed.
        /// Verbose levels are as following (each level contains all of the previous
        /// levels):
        /// * -1 : No message output at all.
        /// *  0 : Default level: Regular output and errors.
        /// *  1 : Warnings.
        /// *  2 : Status messages.
        /// *  3 : Detailed status and debug messages.
        /// *  4 : Very detailed status and debug messages.
        OPI_API_EXPORT static void setVerboseLevel(int level);

        /// \brief Set the message prefix.
        ///
        /// By default, all messages are prepended by a prefix which is initially set to
        /// "OPI". Prefixes appear in rectangular brackets in front of each line.
        /// Use this function to change the prefix to something different. If the new
        /// prefix is empty (""), the brackets are omitted.
        /// \param newPrefix A string to be prepended to all future messages.
        OPI_API_EXPORT static void setPrefix(const char* newPrefix);

        /// \brief Get the current verbose level.
        ///
        /// \returns An integer stating the current verbose level (typically between
        /// -1 and 4; see setVerboseLevel() for details).
        OPI_API_EXPORT static int  getVerboseLevel();

        /// \brief Get the contents of the message buffer and clear it.
        ///
        /// If output mode is set to BUFFER via the setMode() function, messages are
        /// not written but saved in a buffer instead. This function will retrieve
        /// the contents of the buffer, then clear it.
        /// \returns A string containing all buffered messages since the last call to
        /// this function.
        OPI_API_EXPORT static int getBuffer(char*& buffer);

        /// \brief Return a pointer to the current output stream.        
        OPI_API_EXPORT static std::ostream* getStream();

	private:

        // A dummy stream that is used to discard messages.
        struct nullstream: std::ostream
        {
            nullstream(): std::ostream(0) { }
        };

		// Verbose level rule of thumb:
		// Level -1: Quiet
		// Level 0: Always print message (e.g. errors; default)
		// Level 1: Warnings
		// Level 2: Status messages
		// Level 3: Detailed status messages
		// Level 4: Very detailed status messages
		static int verboseLevel;
        static std::string prefix;
        static std::ofstream logFile;
        static std::stringstream messageBuffer;
        static std::ostream* redirect;
        static mode currentMode;
};

}
#endif
