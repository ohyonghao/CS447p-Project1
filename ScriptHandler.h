///////////////////////////////////////////////////////////////////////////////
//
//      ScriptHandler.h                         Author:     Eric McDaniel
//                                              Modified:   Stephen Chenney
//                                              Date:       Fall 2004
//
//      Implementation of CScripthandler methods.  You should not need to
//  modify this file.
//
///////////////////////////////////////////////////////////////////////////////


#ifndef C_SCRIPT_HANDLER
#define C_SCRIPT_HANDLER
#include <string>
class TargaImage;

class CScriptHandler
{
    // methods
    public:
        ///////////////////////////////////////////////////////////////////////////////
        //
        //      Execute the given command string on the given image.  If the command
        //  string could not be parsed, an error message is displayed and false is 
        //  returned.  Otherwise return true.
        //  
        ///////////////////////////////////////////////////////////////////////////////
        static bool HandleCommand(const std::string sCommand, TargaImage*& pImage);

        ///////////////////////////////////////////////////////////////////////////////
        //
        //      The given script file is executed on the given image.  If the file is 
        //  not correctly parsed an error message is printed and false is returned.  
        //  Otherwise if all commands in the script execute correctly true is returned,
        //  otherwise false is returned.
        //
        ///////////////////////////////////////////////////////////////////////////////
        static bool HandleScriptFile(const std::string sFilename, TargaImage*& pImage);
};// CScriptHandler

#endif // _C_SCRIPT_HANDLER


