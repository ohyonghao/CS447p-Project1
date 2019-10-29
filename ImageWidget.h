///////////////////////////////////////////////////////////////////////////////
//
//      ImageWidget.h                           Author:     Stephen Chenney
//                                              Modified:   Eric McDaniel
//                                              Date:       Spring 2002
//
//      Widget to display image and enter commands to modify image.  You do not
//  need to modify this file.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef IMAGE_WIDGET_H
#define IMAGE_WIDGET_H

#include <FL/Fl.H>
#include <FL/Fl_Widget.H>
#include <string>

class Fl_Box;
class Fl_Input;
class TargaImage;

class ImageWidget : public Fl_Widget
{
    // methods
    public:
        ImageWidget(int, int, int, int, char*);
        ~ImageWidget();

	    void draw();	                    // FLTK draw function draws the current image.
	    TargaImage* Get_Image();            // get the current image
	    void Redraw();                      // redraw the image in the window


    private:
        static void CommandCallback(Fl_Widget* pWidget, void* pData);           // command entered callback


    // members
    private:
        TargaImage* m_pImage;	                // The image to display (current image).
        Fl_Box*     m_pStaticTextBox;           // static text
        Fl_Input*   m_pCommandInput;            // input box
};


#endif


