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
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <valarray>
#include <random>
#include <map>
#include <unordered_map>
#include <functional>

using namespace std;

// Computes n choose s, efficiently
constexpr double Binomial(int n, int s)
{
    double        res{1};

    for (int i = 1 ; i <= s ; i++)
        res = (n - i + 1) * res / i ;

    return res;
}// Binomial

// Fills matrix with binomial using an inplace calculation of
// pascal's triangle and then backfilling the values.
template <typename T>
void GaussMask(valarray<T> &matrix){
    const size_t size = static_cast<size_t>(sqrt(matrix.size()));
    const size_t half = (size >> 1 ) ;//+ (size &0b1 ? 1 : 0);
    // If matrix is empty or not square then return
    if( size == 0 || size * size != matrix.size() ){
        return;
    }

    // Calculate sizeth row of pascals triangle
    matrix[0] = 1;
    for( size_t i = 1; i < size; ++i){
        matrix[i*size] = matrix[i*size+i] = 1;
        for( size_t j = 1; j < i; ++j){
            matrix[i*size+j] = matrix[(i-1)*size + j - 1] + matrix[(i-1)*size + j];
        }
    }
    // Fill in matrix
    for( size_t i = 1; i < size - 1; ++i ){
        // Fill in edge
        matrix[i] = matrix[i*size] = matrix[(size-1)*size + i];
        // Now fill in middle diagonally
        for( size_t j = 1; j <= i; ++j){
            matrix[i*size + j] = matrix[j] * matrix[i*size];
        }
    }
    // Backfill
    for( size_t i = size - 2; i != 0; --i){
        // Fill in size - i on the end
        if( i > half ){
            for( size_t j = 0; j < size - i; ++j){
                matrix[(i+1)*size - j - 1] = matrix[i*size + j];
            }
        }else if( i < half ){
            for( size_t j = 0; j < size - i; ++j){
                matrix[(i+1)*size - j - 1] = matrix[(i+half)*size + j];
            }
        }else{
            for( size_t j = 0; j < half; ++j){
                matrix[(i+1)*size - j - 1] = matrix[i*size + j];
            }
        }
    }
    matrix[size-1] = 1; // Fix the top right corner
}
template< int N, int M, typename IN, typename OUT, typename UNOP>
void transform_n_less_m(IN first, IN last, OUT result, UNOP op){
    while( first != last ){
        *result = op(*first);
        for( auto i = 1; i < N - M; ++i){
            *(result+i) = *result;
        }
        result+=N; first+=N;
    }
}

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
    data(static_cast<size_t>(w*h*4))
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
    data.resize(static_cast<size_t>(_width * _height * 4));

    copy(d, d + data.size(), data.begin());
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
    cout << endl;

    return rgb;
}// TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Save the image to a targa file. Returns 1 on success, 0 on failure.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Save_Image(const string filename)
{
    TargaImage	*out_image = Reverse_Rows();

    if (! out_image)
	    return false;

    if (!tga_write_raw(filename.data(), _width, _height, out_image->data.data(), TGA_TRUECOLOR_32))
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
TargaImage* TargaImage::Load_Image(const string filename)
{
    unsigned char   *temp_data;
    TargaImage	    *temp_image;
    TargaImage	    *result;
    int		        width{0}, height{0}; // Initialize to sensible values

    if (filename.empty())
    {
        cout << "No filename given." << endl;
        return nullptr;
    }// if

    temp_data = reinterpret_cast<unsigned char*>(tga_load(filename.data(), &width, &height, TGA_TRUECOLOR_32));
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
    for( auto it = data.begin(); it < data.end(); it+=4){
        this->RGBA_To_RGB(it, it); // overwriting itself
        uchar y = static_cast<uchar>(
                  *(it+RED)   * 0.299
                + *(it+GREEN) * 0.587
                + *(it+BLUE)  * 0.114);
        *(it+RED)   = y;
        *(it+GREEN) = y;
        *(it+BLUE)  = y;
    }
    return true;
}// To_Grayscale


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using uniform quantization.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////

bool TargaImage::Quant_Uniform(uchar r, uchar g, uchar b){
    // Clear the n bits, then move it to the midpoint of each bucket
    int count{-1}; // used statically in transform

    transform(data.begin(),data.end(),data.begin(),[=,&count](auto c){
        ++count;
        auto quant = [c](auto shift)->uchar{
            return static_cast<uchar>(((c >> (8 - shift) ) << (8 - shift)) + (1<<(8 - (shift + 1))));
        };
        switch( count ){
        case RED:
            return quant(r);
        case GREEN:
            return quant(g);
        case BLUE:
            return quant(b);
        case ALPHA:
        default:
            count = -1;
            return c;
        }

    });
    return true;
}

bool TargaImage::Quant_Uniform()
{
    // Should be a simple transform
    // starting at -1 is so we can increment first and it makes the
    // switch easy

    Quant_Uniform(3,3,2);
    return true;
}// Quant_Uniform


///////////////////////////////////////////////////////////////////////////////
//
//      Convert the image to an 8 bit image using populosity quantization.  
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Quant_Populosity()
{
    // Endian issue
    auto unmask = [](uint32_t rgb)->tuple<uchar,uchar,uchar>{
        //
        uchar r = (rgb) & 0b11111111;
        uchar g = (rgb >> 8) & 0b11111111;
        uchar b = (rgb >> 16 ) & 0b11111111;
        return {r,g,b};
    };

    // We want to convert from 24bit to 8bit

    // Do a Quant_Uniform down to 5 for each level (5x5x5) rather than
    // what we did for Quant_Uniform of (3x3x2) (these are powers of 2)
    Quant_Uniform(5,5,5);
    unordered_map<uint32_t,uint64_t> count_sort;

    // find the 256 most popular colors
    // I need to get this down to only the 2^15 colors
    // so I need to convert to a unordered_map most likely
    // so I can minimize the number of items, this has the
    // added benefit of then not needing to check for 0 in
    // my color_map step
    for( auto it = data.begin(); it < data.end(); it+=4 ){
        // Read in all colors and remove alpha
        uint32_t c = *reinterpret_cast<uint32_t*>(&it);
        count_sort[c]++;
    }
    // This is all wrong
    typedef pair<uint32_t,uint64_t> key_pair;
    vector<key_pair> sorted;
    sorted.reserve(1<<15);
    for( auto it = count_sort.begin(); it != count_sort.end(); ++it ){
        sorted.push_back(*it);
    }
    // Only sort out the top 256, then quit
    partial_sort(sorted.begin(),sorted.begin()+256,sorted.end(),[](auto lhs, auto rhs)->bool{
        return lhs.second > rhs.second;
    });

    // We're going to dp this, make a mapping from color->quantized
    unordered_map<uint32_t,uint32_t> color_map;
    for_each( count_sort.begin(), count_sort.end(), [&](auto cs){
        auto t = unmask(cs.first);
        uchar r1 = get<RED>(t);
        uchar g1 = get<GREEN>(t);
        uchar b1 = get<BLUE>(t);
        uint32_t nearest_color{0};
        uint32_t closest_distance = numeric_limits<uint32_t>::max();
        // Brute force finding it through the first 256 in sorted
        for_each(sorted.begin(),sorted.begin()+256,[&](auto s){
            auto [r2,g2,b2] = unmask(s.first);
            uint32_t distance = (r1-r2)*(r1-r2) + (g1-g2)*(g1-g2) + (b1-b2)*(b1-b2);
            if( distance < closest_distance ){
                nearest_color = s.first;
                closest_distance = distance;
            }
        });
        color_map[cs.first] = nearest_color;
    });
    for( auto it = data.begin(); it < data.end(); it+=4 ){
        if( color_map.count(*reinterpret_cast<uint32_t*>(&it)) == 0 ){
            cout << "Uh-oh, something went wrong, a color is missing" << endl;
        }
        *reinterpret_cast<uint32_t*>(&it) = color_map[*reinterpret_cast<uint32_t*>(&it)];
    }
    // Now, for each pixel we calculate the distance
    return true;
}// Quant_Populosity


///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image using a threshold of 1/2.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Threshold()
{
    // No need to convert to float first, as we can calculate the
    // midpoint as 256/2 = 128.
    this->To_Grayscale();
    transform_n_less_m<4,1>(data.begin(), data.end(), data.begin(), [](auto c)->uchar{
        return (c<128) ? 0 : 255;
    });
    return true;
}// Dither_Threshold


///////////////////////////////////////////////////////////////////////////////
//
//      Dither image using random dithering.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Random()
{
    // Still pointless to convert to float, we can choose a random value
    // from the range of 0-255 since a conversion to float and back to
    // uchar will be the same as not having converted in the first place.

    // Think about this a little more, what is the algoritm trying to do?
    // We want to add a random amount from [-a,a], then apply the threshold
    // algorithm to it.
    this->To_Grayscale();
    constexpr int32_t a = 51; // An artistic value - should be close to [-0.2,0.2]
    default_random_engine generator;
    uniform_int_distribution<int32_t> distribution(-a,a);
    auto randint = bind(distribution,generator );

    transform_n_less_m<4,1>(data.begin(), data.end(), data.begin(), [&randint](auto c)->uchar{
        // First we check if the noise will bring us outside the range
        // of the char to protect against overflow.
        auto noise = randint();
        if( noise < 0 ){
            c = (c < -noise) ? 0 : c + static_cast<uchar>(-noise);
        }else{
            c = (255 - noise < c) ? 255 : c + static_cast<uchar>(noise);
        }
        return (c < 128) ? 0 : 255;
    });
    return true;
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
    // If the image is so large we can't hold its sum in a uint64_t
    if( numeric_limits<uint64_t>::max()/(static_cast<uint32_t>(_width)*8) < static_cast<uint32_t>(_height) ){
        cout << "Error: Image too large for uint64_t" << endl;
        return false;
    }
    // No need to convert to float first, we'll calculate the average
    this->To_Grayscale();

    // 32bit only allows for 4000x4000 image, 2^63 should be enough
    // as this allows for 3e9 square images.
    uint64_t intensity{0};
    vector<uint64_t> count_sort(256); // initialized to 0
    for( auto it = data.begin(); it < data.end(); it+=4){
        count_sort[*it]++;
        intensity += *it;
    }
    int64_t intensity_index = intensity >> 8;

    // Count down the threshold as these are the values that
    // determine brightness
    uchar threshold{255};

    for(; intensity_index > 0; --threshold){
        intensity_index -= count_sort[threshold];
    }
    // We overshoot by 1 each time
    ++threshold;

    transform_n_less_m<4,1>(data.begin(), data.end(), data.begin(), [threshold](auto c)->uchar{
        return (c<threshold) ? 0 : 255;
    });
    return true;
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

bool TargaImage::Difference(const TargaImage& pImage)
{
    if (_width != pImage._width || _height != pImage._height)
    {
        cout << "Difference: Images not the same size\n";
        return false;
    }// if

    return Difference(pImage.data);
}
bool TargaImage::Difference(const vector<uchar>& remove )
{
    // Can only check if same number of pixels, can't stop
    // all the stupid.
    if( data.size() != remove.size() ){
        cout << "Difference: Images not the same size\n";
        return false;
    }
    auto rgba1 = data.begin();
    auto rgba2 = remove.begin();
    for ( ; rgba1 < data.end() ; rgba1 += 4, rgba2 +=4)
    {
        // Need to do the conversion on the fly for RGBA to RGB...
        // Need to split the function
        vector<uchar> rgb1(3);
        vector<uchar> rgb2(3);
        RGBA_To_RGB(rgba1, rgb1.begin());
        RGBA_To_RGB(rgba2, rgb2.begin());

        *(rgba1 + RED)   = static_cast<uchar>(abs(rgb1[RED]   - rgb2[RED] ));
        *(rgba1 + GREEN) = static_cast<uchar>(abs(rgb1[GREEN] - rgb2[GREEN] ));
        *(rgba1 + BLUE)  = static_cast<uchar>(abs(rgb1[BLUE]  - rgb2[BLUE] ));
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
    const valarray<uint32_t> matrix= {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    return Apply_Mask<uint32_t,uint32_t>(matrix, [](auto c)->uint32_t{return c/9;});
}// Filter_Box


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Bartlett filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Bartlett()
{
    const valarray<uint32_t> matrix= {
        1, 2, 3, 2, 1,
        2, 4, 6, 4, 2,
        3, 6, 9, 6, 3,
        2, 4, 6, 4, 2,
        1, 2, 3, 2, 1
    };
    return Apply_Mask<uint32_t,uint32_t>(matrix, [](auto c)->uint32_t{return c/81;});
}// Filter_Bartlett


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Gaussian()
{
    return Filter_Gaussian_N(5);
}// Filter_Gaussian

///////////////////////////////////////////////////////////////////////////////
//
//      Perform NxN Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////

bool TargaImage::Filter_Gaussian_N( unsigned int N )
{
    valarray<uint32_t> matrix(N*N);
    GaussMask(matrix);

    // The division is 1/(2^{(n-1)*2}
    // That gets us 3=1/16, 5=1/256, 7=1/1024, etc.
    return Apply_Mask<uint32_t,uint32_t>(matrix, [N](auto c){return c >> ((N-1)*2);});
}// Filter_Gaussian_N


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 edge detect (high pass) filter on this image.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Edge()
{
    valarray<int64_t> matrix(5*5);
    GaussMask(matrix);
    matrix *= -1;
    matrix[12] = (256-36);

    Apply_Mask<int64_t,uint64_t>(matrix, [](int64_t c)->int64_t{
        return c < 0 ? 0 : c >> 8;
    });

    return true;
}// Filter_Edge


///////////////////////////////////////////////////////////////////////////////
//
//      Perform a 5x5 enhancement filter to this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Enhance()
{
    valarray<int64_t> matrix(5*5);
    GaussMask(matrix);
    matrix*=-1;
    matrix[12] = 2*256-36;

    Apply_Mask<int64_t,uint64_t>(matrix, [](int64_t c)->int64_t{
        return c < 0 ? 0 : c >> 8;
    });
    return true;
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
bool TargaImage::Resize(float /*scale*/)
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
bool TargaImage::Rotate(float /*angleDegrees*/)
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
        for (int i = 0 ; i < 3 ; ++i)
            *(out+i) = static_cast<uchar>(clamp( static_cast<int>(floor(*(in+i)) * 255.0 / alpha), 0, 255));
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

    for (uint32_t i = 0 ; i < static_cast<uint32_t>(_height) ; i++)
    {
        uint32_t in_offset = (static_cast<uint32_t>(_height) - i - 1) * static_cast<uint32_t>(_width) * 4;
        uint32_t out_offset = i * static_cast<uint32_t>(_width) * 4;

        for (uint32_t j = 0 ; j < static_cast<uint32_t>(_width) ; j++)
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
   int radius_squared = static_cast<int>(s.radius) * static_cast<int>(s.radius);
   for (int x_off = -(static_cast<int>(s.radius)); x_off <= static_cast<int>(s.radius); x_off++) {
      for (int y_off = -(static_cast<int>(s.radius)); y_off <= static_cast<int>(s.radius); y_off++) {
         int x_loc = static_cast<int>(s.x) + x_off;
         int y_loc = static_cast<int>(s.y) + y_off;
         // are we inside the circle, and inside the image?
         if ((x_loc >= 0 && x_loc < _width && y_loc >= 0 && y_loc < _height)) {
            int dist_squared = x_off * x_off + y_off * y_off;
            if (dist_squared <= radius_squared) {
               data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 0)] = s.r;
               data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 1)] = s.g;
               data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 2)] = s.b;
               data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 3)] = s.a;
            } else if (dist_squared == radius_squared + 1) {
               data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 0)] =
                  (data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 0)] + s.r) / 2;
               data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 1)] =
                  (data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 1)] + s.g) / 2;
               data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 2)] =
                  (data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 2)] + s.b) / 2;
               data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 3)] =
                  (data[static_cast<size_t>((y_loc * _width + x_loc) * 4 + 3)] + s.a) / 2;
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

