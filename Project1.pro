TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += x11


#------------------------------------------------
#
# Uso librerie FLTK 1.3.1
#
#------------------------------------------------


INCLUDEPATH += /usr/local/include

LIBS += /usr/local/lib/libfltk.a \
    /usr/local/lib/libfltk_forms.a \
    /usr/local/lib/libfltk_gl.a \
    /usr/local/lib/libfltk_images.a


SOURCES += \
        ImageWidget.cpp \
        Main.cpp \
        ScriptHandler.cpp \
        TargaImage.cpp \
        libtarga.c

HEADERS += \
    Globals.h \
    ImageWidget.h \
    ScriptHandler.h \
    TargaImage.h \
    libtarga.h

# X11
unix|win32: LIBS += -ldl -lXft -lfontconfig -lXrender -lXfixes -lXinerama -lXcursor

# OpenGL
unix|win32: LIBS += -lGL -lGLU
