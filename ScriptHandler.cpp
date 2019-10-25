///////////////////////////////////////////////////////////////////////////////
//
//      ScriptHandler.cpp                       Author:     Eric McDaniel
//                                              Modified:   Stephen Chenney
//                                              Date:       Fall 2004, Sept 29
//
//      Implementation of CScripthandler methods.  You should not need to
//  modify this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "Globals.h"
#include "ScriptHandler.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <sstream>
#include <charconv>
#include "TargaImage.h"

using namespace std;

// constants
const int       c_maxLineLength         = 1000;
const vector<string>      c_asCommands  = {
                        "load",                     // valid commands
                        "save",
                        "run",
                        "gray",
                        "quant-unif",
                        "quant-pop",
                        "dither-thresh",
                        "dither-rand",
                        "dither-fs",
                        "dither-bright",
                        "dither-cluster",
                        "dither-pattern",
                        "dither-color",
                        "filter-box",
                        "filter-bartlett",
                        "filter-gauss",
                        "filter-gauss-n",
                        "filter-edge",
                        "filter-enhance",
                        "npr-paint",
                        "half",
                        "double",
                        "scale",
                        "comp-over",
                        "comp-in",
                        "comp-out",
                        "comp-atop",
                        "comp-xor",
                        "diff",
                        "rotate"
                      };

enum ECommands          // command ids
{
    LOAD,
    SAVE,
    RUN,
    GREY,
    QUANT_UNIF,
    QUANT_POP,
    DITHER_THRESH,
    DITHER_RAND,
    DITHER_FS,
    DITHER_BRIGHT,
    DITHER_CLUSTER,
    DITHER_PATTERN,
    DITHER_COLOR,
    FILTER_BOX,
    FILTER_BARTLETT,
    FILTER_GAUSS,
    FILTER_GAUSS_N,
    FILTER_EDGE,
    FILTER_ENHANCE,
    NPR_PAINT,
    HALF,
    DOUBLE,
    SCALE,
    COMP_OVER,
    COMP_IN,
    COMP_OUT,
    COMP_ATOP,
    COMP_XOR,
    DIFF,
    ROTATE,
    NUM_COMMANDS
};// ECommands


///////////////////////////////////////////////////////////////////////////////
//
//      Execute the given command string on the given image.  If the command
//  string could not be parsed, an error message is displayed and false is 
//  returned.  Otherwise return true.
//  
///////////////////////////////////////////////////////////////////////////////
bool CScriptHandler::HandleCommand(const string sCommand, TargaImage*& pImage)
{
    if (sCommand.empty()){
        return true;
    }
	string sCommandLine{sCommand};
	std::istringstream iss(sCommandLine);
	string sToken;
	iss >> sToken;

    // find command that was given
    size_t command;
    for (command = 0; command < NUM_COMMANDS; ++command)
        if( sToken == c_asCommands[command])
            break;

    // if there's no image only a subset of commands are valid
    if (!pImage && command != LOAD && command != RUN && command != NUM_COMMANDS)
    {
        cout << "No image to operate on.  Use \"load\" command to load image." << endl;
        return false;
    }// if

    // handle the command
    bool bResult,
         bParsed = true;

    switch (command)
    {
        case LOAD:
        {
            if (pImage)
                delete pImage;
			string sFilename;
			iss >> sFilename;
            bResult = (pImage = TargaImage::Load_Image(sFilename)) != nullptr;

            if (!bResult)
            {
                if (sFilename.empty())
                    cout << "Unable to load image:  " << endl;
                else
                    cout << "Unable to load image:  " << sFilename << endl;
                
                bParsed = false;
            }// if
            break;
        }// LOAD

        case SAVE:
        {
			string sFilename;
			iss >> sFilename;
            if (sFilename.empty())
                cout << "No filename given." << endl;

            bParsed = !sFilename.empty();
            bResult =  bParsed && pImage->Save_Image(sFilename);
            break;
        }// SAVE

        case RUN:
        {
			string runcommand;
			iss >> runcommand;
            bResult = HandleScriptFile(runcommand.c_str(), pImage);
            break;
        }// RUN

        case GREY:
        {
            bResult = pImage->To_Grayscale();
            break;
        }// GREY

        case QUANT_UNIF:
        {
            bResult = pImage->Quant_Uniform();
            break;
        }// QUANT_UNIF

        case QUANT_POP:
        {
            bResult = pImage->Quant_Populosity();
            break;
        }// QUANT_POP

        case DITHER_THRESH:
        {
            bResult = pImage->Dither_Threshold();
            break;
        }// QUANT_THRESH

        case DITHER_RAND:
        {
            bResult = pImage->Dither_Random();
            break;
        }// DITHER_RAND

        case DITHER_FS:
        {
            bResult = pImage->Dither_FS();
            break;
        }// DITHER_FS

        case DITHER_BRIGHT:
        {
            bResult = pImage->Dither_Bright();
            break;
        }// DITHER_BRIGHT
        
        case DITHER_CLUSTER:
        {
            bResult = pImage->Dither_Cluster();
            break;
        }// DITHER_CLUSTER
        
        case DITHER_COLOR:
        {
            bResult = pImage->Dither_Color();
            break;
        }// DITHER_COLOR

        case FILTER_BOX:
        {
            bResult = pImage->Filter_Box();
            break;
        }// DITHER_BOX

        case FILTER_BARTLETT:
        {
            bResult = pImage->Filter_Bartlett();
            break;
        }// DITHER_BARTLETT

        case FILTER_GAUSS:
        {
            bResult = pImage->Filter_Gaussian();
            break;
        }// FILTER_GUASS

        case FILTER_GAUSS_N:
        {
			string sN;
			iss >> sN;
            
            int N;
            from_chars(sN.data(),sN.data()+sN.size(),N);
            if (N % 2 != 1 || N < 0) {
               cout << "N \"" << N << "\" is not allowed; N must be an odd number." << endl;
               break;
            }
            bResult = pImage->Filter_Gaussian_N(static_cast<uint32_t>(N));
            break;
        }// FILTER_GUASS_N

        case FILTER_EDGE:
        {
            bResult = pImage->Filter_Edge();
            break;
        }// FILTER_EDGE

        case FILTER_ENHANCE:
        {
            bResult = pImage->Filter_Enhance();
            break;
        }// FILTER_ENHANCE

        case NPR_PAINT:
        {
            bResult = pImage->NPR_Paint();
            break;
        }// NPR_PAINT


        case HALF:
        {
            bResult = pImage->Half_Size();
            break;
        }// HALF

        case DOUBLE:
        {
            bResult = pImage->Double_Size();
            break;
        }// DOUBLE

        case SCALE:
        {
			string sScale;
			iss >> sScale;
            
            double scale = atof(sScale.c_str());

            if(scale <= 0)
            {
                cout << "Invalid scaling factor." << endl;
                bParsed = bResult = false;
            }// if
            else
                bResult = pImage->Resize(static_cast<float>(scale));
            break;
        }// SCALE

        case COMP_OVER:
        {
			string sFilename;
			iss >> sFilename;
            TargaImage* pNewImage = TargaImage::Load_Image(sFilename);
            if (!pNewImage)
            {
                if (!sFilename.empty())
                    cout << "Unable to load image:  " << sFilename << endl;
                else
                    cout << "No filename given." << endl;
                bParsed = false;
            }// if
            bResult = pNewImage && pImage->Comp_Over(*pNewImage);
            delete pNewImage;
            break;
        }// COMP_OVER

        case COMP_IN:
        {
			string sFilename;
			iss >> sFilename;
            TargaImage* pNewImage = TargaImage::Load_Image(sFilename);
            if (!pNewImage)
            {
                if (!sFilename.empty())
                    cout << "Unable to load image:  " << sFilename << endl;
                else
                    cout << "No filename given." << endl;

                bParsed = false;
            }// if
            bResult = pNewImage && pImage->Comp_In(*pNewImage);
            delete pNewImage;
            break;
        }// COMP_IN

        case COMP_OUT:
        {
			string sFilename;
			iss >> sFilename;
            TargaImage* pNewImage = TargaImage::Load_Image(sFilename);
            if (!pNewImage)
            {
                if (!sFilename.empty())
                    cout << "Unable to load image:  " << sFilename << endl;
                else
                    cout << "No filename given." << endl;

                bParsed = false;
            }// if
            bResult = pNewImage && pImage->Comp_Out(*pNewImage);
            delete pNewImage;
            break;
        }// COMP_OUT

        case COMP_ATOP:
        {
			string sFilename;
			iss >> sFilename;
            TargaImage* pNewImage = TargaImage::Load_Image(sFilename);
            if (!pNewImage)
            {
                if (!sFilename.empty())
                    cout << "Unable to load image:  " << sFilename << endl;
                else
                    cout << "No filename given." << endl;

                bParsed = false;
            }// if
            bResult = pNewImage && pImage->Comp_Atop(*pNewImage);
            delete pNewImage;
            break;
        }// COMP_ATOP

        case COMP_XOR:
        {
			string sFilename;
			iss >> sFilename;
            TargaImage* pNewImage = TargaImage::Load_Image(sFilename);
            if (!pNewImage)
            {
                if (!sFilename.empty())
                    cout << "Unable to load image:  " << sFilename << endl;
                else
                    cout << "No filename given." << endl;

                bParsed = false;
            }// if
            bResult = pNewImage && pImage->Comp_Xor(*pNewImage);
            delete pNewImage;
            break;
        }// COMP_XOR

        case DIFF:
        {
			string sFilename;
			iss >> sFilename;
            TargaImage* pNewImage = TargaImage::Load_Image(sFilename);
            if (!pNewImage)
            {
                if (!sFilename.empty())
                    cout << "Unable to load image:  " << sFilename << endl;
                else
                    cout << "Unable to load image:  " << endl;

                bParsed = false;
            }// if
            bResult = pNewImage && pImage->Difference(*pNewImage);
            delete pNewImage;
            break;
        }// DIFF

        case ROTATE:
        {
			string sAngle;
			iss >> sAngle;
            float angle{0.0};
            if( !sAngle.empty() ){
                angle = static_cast<float>(atof(sAngle.data()));
            }
            if (angle == 0.0f)
            {
                cout << "Invalid rotation angle." << endl;
                bResult = bParsed = false;
            }// if
            else
                bResult = pImage->Rotate(angle);
            break;
        }// ROTATE

        default:
        {
            cout << "Unable to parse command:  " << sCommand << endl;
            bResult = false;
            bParsed = false;
        }// default
    }// switch
	
    return bParsed;
}// HandleCommand


///////////////////////////////////////////////////////////////////////////////
//
//      The given script file is executed on the given image.  If the file is 
//  not correctly parsed an error message is printed and false is returned.  
//  If all commands in the script execute correctly true is returned,
//  otherwise false is returned.
//
///////////////////////////////////////////////////////////////////////////////
bool CScriptHandler::HandleScriptFile(const string sFilename, TargaImage*& pImage)
{
    if (sFilename.empty())
    {
        cout << "No filename given." << endl;
        return false;
    }// if

    ifstream inFile(sFilename);

    if (!inFile.is_open())
    {
        cout << "Unable to open file:  " << sFilename << endl;
        return false;
    }// if

    bool bResult = true;
    string sLine;
    while (!inFile.eof() && bResult)
    {
        getline(inFile, sLine);
        if (!inFile.eof() && sLine.size() > 1)
            bResult = HandleCommand(sLine, pImage);
    }// while

    inFile.close();
    return bResult;
}// CScriptHandler



