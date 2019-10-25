///////////////////////////////////////////////////////////////////////////////
//
//      TargaImage.h                            Author:     Stephen Chenney
//                                              Modified:   Eric McDaniel
//                                              Date:       Fall 2004
//
//      Class to manipulate targa images.  You must implement the image 
//  modification functions.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef TARGA_IMAGE_H_
#define TARGA_IMAGE_H_

#include <FL/Fl.H>
#include <FL/Fl_Widget.H>
#include <stdio.h>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <valarray>

class Stroke;
class DistanceImage;

class TargaImage
{

    // constants
    static constexpr int           RED             = 0;                // red channel
    static constexpr int           GREEN           = 1;                // green channel
    static constexpr int           BLUE            = 2;                // blue channel
    static constexpr int           ALPHA           = 3;                // alpha channel
    static constexpr unsigned char BACKGROUND[3]   = { 0, 0, 0 };          // background color


    // methods
    public:
	    TargaImage(void);
        TargaImage(int w, int h);
        TargaImage(int w, int h, unsigned char *d);
        TargaImage(int w, int h, const std::vector<uchar>& d);
        TargaImage(const TargaImage& image);
	    ~TargaImage(void);

        std::vector<uchar>	To_RGB();	            // Convert the image to RGB format,
        bool Save_Image(const std::string);               // save the image to a file
        static TargaImage* Load_Image(const std::string);       // Load a file and return a pointer to a new TargaImage object.  Returns NULL on failure

        bool To_Grayscale();

        bool Quant_Uniform(uchar r, uchar g, uchar b);
        bool Quant_Uniform();
        bool Quant_Populosity();
        bool Quant_Median();

        bool Dither_Threshold();
        bool Dither_Random();
        bool Dither_FS();
        bool Dither_Bright();
        bool Dither_Cluster();
        bool Dither_Color();

        bool Comp_Over(const TargaImage &pImage);
        bool Comp_In(const TargaImage& pImage);
        bool Comp_Out(const TargaImage &pImage);
        bool Comp_Atop(const TargaImage &pImage);
        bool Comp_Xor(TargaImage* pImage);

        bool Difference(const TargaImage& pImage);
        bool Difference(const std::vector<uchar> &remove );

        template <typename T, typename S>
        bool Apply_Mask(const std::valarray<T>&, std::function<S(S)>);
        bool Filter_Box();
        bool Filter_Bartlett();
        bool Filter_Gaussian();
        bool Filter_Gaussian_N(unsigned int N);
        bool Filter_Edge();
        bool Filter_Enhance();

        bool NPR_Paint();

        bool Half_Size();
        bool Double_Size();
        bool Resize(float scale);
        bool Rotate(float angleDegrees);

    private:


    // reverse the rows of the image, some targas are stored bottom to top
    TargaImage* Reverse_Rows(void);

	// clear image to all black
        void ClearToBlack();

	// Draws a filled circle according to the stroke data
        void Paint_Stroke(const Stroke& s);

        void Paint_Layer( TargaImage& reference ,uint32_t N );
    // members
    private:
        int		_width;	        // width of the image in pixels
        int		_height;	    // height of the image in pixels
        std::vector<uchar>	data;	    // pixel data for the image, assumed to be in pre-multiplied RGBA format.
        // helper function for format conversion
        void RGBA_To_RGB(decltype (data.cbegin()) in, decltype (data.begin()) out);
        template<class T>
        inline size_t index( T w, T h){ return (h*static_cast<T>(_width) + w)*4 ;}

    public:
            inline int width()const {return _width;}
            inline int height()const {return _height;}
};

class Stroke { // Data structure for holding painterly strokes.
public:
   Stroke(void);
   Stroke(unsigned int radius, unsigned int x, unsigned int y,
          unsigned char r, unsigned char g, unsigned char b, unsigned char a);
   
   // data
   unsigned int radius, x, y;	// Location for the stroke
   unsigned char r, g, b, a;	// Color
};

///////////////////////////////////////////////////////////////////////////////
//
//      My range class
//
///////////////////////////////////////////////////////////////////////////////

template <typename IT>
class range{
private:
    IT first;
    IT last;
public:
    range(IT first,IT last):first{first}, last{last}{}
    range(range&& m):first{m.first},last{m.last}{}
    range(const range& m) = default;
    range& operator=(const range& m) = default;
    IT begin() {return first;}
    IT end() {return last;}
    range& operator++(){++first; ++last; return *this;}
    const range operator++(int){range orig{this}; ++first; ++last; return orig;}
    range& operator+=(int N){first+=N; last+=N; return *this;}
};

///////////////////////////////////////////////////////////////////////////////
//
//      Apply a given matrix to the image
//
///////////////////////////////////////////////////////////////////////////////
template <typename T, typename S>
bool TargaImage::Apply_Mask(const std::valarray<T> &matrix, std::function<S(S)> op){

    auto masked{data};

    std::valarray<T> result[3];
    for( auto& v: result ){
        v.resize(matrix.size());
    }
    int dim = static_cast<int>(sqrt(matrix.size())/2);
    for( int j = 0; j < _height; ++j ){
        for( int i = 0; i < _width; ++i ){
            int xindex = dim;
            int yindex = -dim;
            // Load the matrix
            for( size_t k = 0; k < matrix.size(); ++k ){
                // We'll do it a slow way at first, then think about optimization
                // The biggest roadblock to a good algorithm is optimizing too early
                result[RED]  [k] = data[index(std::clamp(i+xindex, 0, _width-1 ), std::clamp(j+yindex, 0, _height-1) ) + RED];
                result[GREEN][k] = data[index(std::clamp(i+xindex, 0, _width-1 ), std::clamp(j+yindex, 0, _height-1) ) + GREEN];
                result[BLUE] [k] = data[index(std::clamp(i+xindex, 0, _width-1 ), std::clamp(j+yindex, 0, _height-1) ) + BLUE];
                // Update indexes
                if( yindex == dim ){
                    --xindex;
                    yindex = -dim;
                }else{
                    ++yindex;
                }
            } // k

            // hadamard
            for( size_t i = 0; i < 3; ++i ){
                result[i] *= matrix;
            }
            // Now stuff it back in
            masked[ index(i,j) + RED ]   = std::clamp( op((result[RED]  .sum()) ), static_cast<S>(0), static_cast<S>(255u));
            masked[ index(i,j) + GREEN ] = std::clamp( op((result[GREEN].sum()) ), static_cast<S>(0), static_cast<S>(255u));
            masked[ index(i,j) + BLUE ]  = std::clamp( op((result[BLUE] .sum()) ), static_cast<S>(0), static_cast<S>(255u));
            // Update indexes
        } // j
    } // i

    swap(data,masked);
    return true;
}
#endif


