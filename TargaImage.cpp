///////////////////////////////////////////////////////////////////////////////
//
//      TargaImage.cpp                          Author:     Stephen Chenney
//                                              Modified:   Eric McDaniel
//                                              Date:       Fall 2004
//                                              Modified:   Feng Liu
//                                              Date:       Winter 2011
//                                              Why:        Change the library file 
//      Implementation of TargaImage methods.  You must implement the image
//  modification functions.
//
///////////////////////////////////////////////////////////////////////////////

#include "Globals.h"
#include "TargaImage.h"
#include "libtarga.h"
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;

// constants
constexpr int           RED             = 0;                // red channel
constexpr int           GREEN           = 1;                // green channel
constexpr int           BLUE            = 2;                // blue channel
constexpr int           ALPHA           = 3;                // alpha channel
const unsigned char BACKGROUND[3]   = { 0, 0, 0 };          // background color


// Computes n choose s, efficiently
double Binomial(int n, int s)
{
    double        res;

    res = 1;
    for (int i = 1 ; i <= s ; i++)
        res = (n - i + 1) * res / i ;

    return res;
}// Binomial


///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage() :
    _width{0},
    _height{0}
{}// TargaImage

///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h) :
    _width{w},
    _height{h},
    data(w*h*4u)
{
   ClearToBlack();
}// TargaImage



///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables to values given.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h, unsigned char *d) :
    _width{w},
    _height{h}
{
    uint i;

    data.resize(_width * _height * 4u);

    for (i = 0; i < _width * _height * 4u; i++)
	    data[i] = d[i];
}// TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables to values given.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h, const vector<uchar> &d):
    _width{w},
    _height{h},
    data{d}
{
}// TargaImage

///////////////////////////////////////////////////////////////////////////////
//
//      Copy Constructor.  Initialize member to that of input
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(const TargaImage& image) :
    _width{image._width},
    _height{image._height},
    data{image.data}
{
}


///////////////////////////////////////////////////////////////////////////////
//
//      Destructor.  Free image memory.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::~TargaImage()
{
}// ~TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Converts an image to RGB form, and returns the rgb pixel data - 24 
//  bits per pixel. The returned space should be deleted when no longer 
//  required.
//
///////////////////////////////////////////////////////////////////////////////
vector<uchar> TargaImage::To_RGB()
{
    vector<uchar> rgb(static_cast<size_t>(_width * _height * 3));

    if (data.empty())
        return rgb;

    auto it = data.cbegin();
    auto ot = rgb.begin();
    for( ; it < data.cend(); it+=4, ot+=3 ){
        RGBA_To_RGB( it, ot );
    }

    return rgb;
}// TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Save the image to a targa file. Returns 1 on success, 0 on failure.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Save_Image(const char *filename)
{
    TargaImage	*out_image = Reverse_Rows();

    if (! out_image)
	    return false;

    if (!tga_write_raw(filename, _width, _height, out_image->data.data(), TGA_TRUECOLOR_32))
    {
        cout << "TGA Save Error: " <<  tga_error_string(tga_get_last_error()) << endl;
	    return false;
    }

    delete out_image;

    return true;
}// Save_Image


///////////////////////////////////////////////////////////////////////////////
//
//      Load a targa image from a file.  Return a new TargaImage object which 
//  must be deleted by caller.  Return nullptr on failure.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage* TargaImage::Load_Image(char *filename)
{
    unsigned char   *temp_data;
    TargaImage	    *temp_image;
    TargaImage	    *result;
    int		        width, height;

    if (!filename)
    {
        cout << "No filename given." << endl;
        return nullptr;
    }// if

    temp_data = reinterpret_cast<unsigned char*>(tga_load(filename, &width, &height, TGA_TRUECOLOR_32));
    if (!temp_data)
    {
        cout << "TGA Error: " << tga_error_string(tga_get_last_error()) << endl;
	    width = height = 0;
        return nullptr;
    }
    temp_image = new TargaImage(width, height, temp_data);
    free(temp_data);

    result = temp_image->Reverse_Rows();

    delete temp_image;

    return result;
}// Load_Image


///////////////////////////////////////////////////////////////////////////////
//
//      Convert image to grayscale.  Red, green, and blue channels should all 
//  contain grayscale value.  Alpha channel shoould be left unchanged.  Return
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::To_Grayscale()
{
    ClearToBlack();
    return false;
}// To_Grayscale


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using uniform quantization.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Quant_Uniform()
{
    ClearToBlack();
    return false;
}// Quant_Uniform


///////////////////////////////////////////////////////////////////////////////
//
//      Convert the image to an 8 bit image using populosity quantization.  
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Quant_Populosity()
{
    ClearToBlack();
    return false;
}// Quant_Populosity


///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image using a threshold of 1/2.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Threshold()
{
    ClearToBlack();
    return false;
}// Dither_Threshold


///////////////////////////////////////////////////////////////////////////////
//
//      Dither image using random dithering.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Random()
{
    ClearToBlack();
    return false;
}// Dither_Random


///////////////////////////////////////////////////////////////////////////////
//
//      Perform Floyd-Steinberg dithering on the image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_FS()
{
    ClearToBlack();
    return false;
}// Dither_FS


///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image while conserving the average brightness.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Bright()
{
    ClearToBlack();
    return false;
}// Dither_Bright


///////////////////////////////////////////////////////////////////////////////
//
//      Perform clustered differing of the image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Cluster()
{
    ClearToBlack();
    return false;
}// Dither_Cluster


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using Floyd-Steinberg dithering over
//  a uniform quantization - the same quantization as in Quant_Uniform.
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Color()
{
    ClearToBlack();
    return false;
}// Dither_Color


///////////////////////////////////////////////////////////////////////////////
//
//      Composite the current image over the given image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Over(TargaImage* pImage)
{
    if (_width != pImage->_width || _height != pImage->_height)
    {
        cout <<  "Comp_Over: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_Over


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image "in" the given image.  See lecture notes for 
//  details.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_In(TargaImage* pImage)
{
    if (_width != pImage->_width || _height != pImage->_height)
    {
        cout << "Comp_In: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_In


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image "out" the given image.  See lecture notes for 
//  details.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Out(TargaImage* pImage)
{
    if (_width != pImage->_width || _height != pImage->_height)
    {
        cout << "Comp_Out: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_Out


///////////////////////////////////////////////////////////////////////////////
//
//      Composite current image "atop" given image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Atop(TargaImage* pImage)
{
    if (_width != pImage->_width || _height != pImage->_height)
    {
        cout << "Comp_Atop: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_Atop


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image with given image using exclusive or (XOR).  Return
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Xor(TargaImage* pImage)
{
    if (_width != pImage->_width || _height != pImage->_height)
    {
        cout << "Comp_Xor: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_Xor


///////////////////////////////////////////////////////////////////////////////
//
//      Calculate the difference bewteen this imag and the given one.  Image 
//  dimensions must be equal.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Difference(TargaImage* pImage)
{
    if (!pImage)
        return false;

    if (_width != pImage->_width || _height != pImage->_height)
    {
        cout << "Difference: Images not the same size\n";
        return false;
    }// if


    for (auto rgba1 = data.begin(), rgba2 = pImage->data.begin() ; rgba1 < data.end() ; rgba1 += 4, rgba2 +=4)
    {
        // Need to do the conversion on the fly for RGBA to RGB...
        // Need to split the function
        vector<uchar> rgb1(3);
        vector<uchar> rgb2(3);
        RGBA_To_RGB(rgba1, rgb1.begin());
        RGBA_To_RGB(rgba2, rgb2.begin());

        *(rgba1 + RED)   = static_cast<uchar>(abs(rgb1[RED]   - rgba2[RED] ));
        *(rgba1 + GREEN) = static_cast<uchar>(abs(rgb1[GREEN] - rgba2[GREEN] ));
        *(rgba1 + BLUE)  = static_cast<uchar>(abs(rgb1[BLUE]  - rgba2[BLUE] ));
        *(rgba1 + ALPHA) = 255;
    }

    return true;
}// Difference


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 box filter on this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Box()
{
    ClearToBlack();
    return false;
}// Filter_Box


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Bartlett filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Bartlett()
{
    ClearToBlack();
    return false;
}// Filter_Bartlett


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Gaussian()
{
    ClearToBlack();
    return false;
}// Filter_Gaussian

///////////////////////////////////////////////////////////////////////////////
//
//      Perform NxN Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////

bool TargaImage::Filter_Gaussian_N( unsigned int /*N*/ )
{
    ClearToBlack();
   return false;
}// Filter_Gaussian_N


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 edge detect (high pass) filter on this image.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Edge()
{
    ClearToBlack();
    return false;
}// Filter_Edge


///////////////////////////////////////////////////////////////////////////////
//
//      Perform a 5x5 enhancement filter to this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Enhance()
{
    ClearToBlack();
    return false;
}// Filter_Enhance


///////////////////////////////////////////////////////////////////////////////
//
//      Run simplified version of Hertzmann's painterly image filter.
//      You probably will want to use the Draw_Stroke funciton and the
//      Stroke class to help.
// Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::NPR_Paint()
{
    ClearToBlack();
    return false;
}



///////////////////////////////////////////////////////////////////////////////
//
//      Halve the dimensions of this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Half_Size()
{
    ClearToBlack();
    return false;
}// Half_Size


///////////////////////////////////////////////////////////////////////////////
//
//      Double the dimensions of this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Double_Size()
{
    ClearToBlack();
    return false;
}// Double_Size


///////////////////////////////////////////////////////////////////////////////
//
//      Scale the image dimensions by the given factor.  The given factor is 
//  assumed to be greater than one.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Resize(float scale)
{
    ClearToBlack();
    return false;
}// Resize


//////////////////////////////////////////////////////////////////////////////
//
//      Rotate the image clockwise by the given angle.  Do not resize the 
//  image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Rotate(float angleDegrees)
{
    ClearToBlack();
    return false;
}// Rotate


//////////////////////////////////////////////////////////////////////////////
//
//      Given a single RGBA pixel return, via the second argument, the RGB
//      equivalent composited with a black background.
//
///////////////////////////////////////////////////////////////////////////////

void TargaImage::RGBA_To_RGB(decltype (data.cbegin()) in, decltype (data.begin()) out){
    auto alpha = *(in + ALPHA);

    if (alpha == 0)
    {
        *(out + RED)   = BACKGROUND[RED];
        *(out + GREEN) = BACKGROUND[GREEN];
        *(out + BLUE)  = BACKGROUND[BLUE];
    }
    else
    {
        double	alpha_scale = 255.0 / alpha;

        for (int i = 0 ; i < 3 ; ++i)
        {
            int val = static_cast<int>(floor(*(in+i)) * alpha_scale);
            if (val < 0)
             *out = 0;
            else if (val > 255)
             *out = 255;
            else
             *out = static_cast<uchar>(val);

            ++out;
        }
    }
}// RGA_To_RGB


///////////////////////////////////////////////////////////////////////////////
//
//      Copy this into a new image, reversing the rows as it goes. A pointer
//  to the new image is returned.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage* TargaImage::Reverse_Rows(void)
{
    vector<uchar>   dest(data.size());

    if (data.empty())
        return nullptr;

    for (int i = 0 ; i < _height ; i++)
    {
        int in_offset = (_height - i - 1) * _width * 4;
        int out_offset = i * _width * 4;

        for (int j = 0 ; j < _width ; j++)
        {
            dest[out_offset + j * 4 + RED]   = data[in_offset + j * 4 + RED];
            dest[out_offset + j * 4 + GREEN] = data[in_offset + j * 4 + GREEN];
            dest[out_offset + j * 4 + BLUE]  = data[in_offset + j * 4 + BLUE];
            dest[out_offset + j * 4 + ALPHA] = data[in_offset + j * 4 + ALPHA];
        }
    }

    return new TargaImage(_width, _height, dest);
}// Reverse_Rows


///////////////////////////////////////////////////////////////////////////////
//
//      Clear the image to all black.
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::ClearToBlack()
{
    transform(data.begin(), data.end(), data.begin(), [](auto &){return 0;});
}// ClearToBlack


///////////////////////////////////////////////////////////////////////////////
//
//      Helper function for the painterly filter; paint a stroke at
// the given location
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::Paint_Stroke(const Stroke& s) {
   int radius_squared = (int)s.radius * (int)s.radius;
   for (int x_off = -((int)s.radius); x_off <= (int)s.radius; x_off++) {
      for (int y_off = -((int)s.radius); y_off <= (int)s.radius; y_off++) {
         int x_loc = (int)s.x + x_off;
         int y_loc = (int)s.y + y_off;
         // are we inside the circle, and inside the image?
         if ((x_loc >= 0 && x_loc < _width && y_loc >= 0 && y_loc < _height)) {
            int dist_squared = x_off * x_off + y_off * y_off;
            if (dist_squared <= radius_squared) {
               data[(y_loc * _width + x_loc) * 4 + 0] = s.r;
               data[(y_loc * _width + x_loc) * 4 + 1] = s.g;
               data[(y_loc * _width + x_loc) * 4 + 2] = s.b;
               data[(y_loc * _width + x_loc) * 4 + 3] = s.a;
            } else if (dist_squared == radius_squared + 1) {
               data[(y_loc * _width + x_loc) * 4 + 0] =
                  (data[(y_loc * _width + x_loc) * 4 + 0] + s.r) / 2;
               data[(y_loc * _width + x_loc) * 4 + 1] =
                  (data[(y_loc * _width + x_loc) * 4 + 1] + s.g) / 2;
               data[(y_loc * _width + x_loc) * 4 + 2] =
                  (data[(y_loc * _width + x_loc) * 4 + 2] + s.b) / 2;
               data[(y_loc * _width + x_loc) * 4 + 3] =
                  (data[(y_loc * _width + x_loc) * 4 + 3] + s.a) / 2;
            }
         }
      }
   }
}


///////////////////////////////////////////////////////////////////////////////
//
//      Build a Stroke
//
///////////////////////////////////////////////////////////////////////////////
Stroke::Stroke() {}

///////////////////////////////////////////////////////////////////////////////
//
//      Build a Stroke
//
///////////////////////////////////////////////////////////////////////////////
Stroke::Stroke(unsigned int iradius, unsigned int ix, unsigned int iy,
               unsigned char ir, unsigned char ig, unsigned char ib, unsigned char ia) :
   radius(iradius),x(ix),y(iy),r(ir),g(ig),b(ib),a(ia)
{
}

