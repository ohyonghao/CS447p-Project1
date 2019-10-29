///////////////////////////////////////////////////////////////////////////////
//
//      Globals.inl                                 Author:  Eric McDaniel
//                                                  Email:   chate@cs.wisc.edu
//                                                  Date:    Summer 2001
//
//      Global inline methods and templates.
//
///////////////////////////////////////////////////////////////////////////////

#include <functional>


///////////////////////////////////////////////////////////////////////////////
//
//      Get the minimum of two values.  Note:  operator < must be defined for
//  the given type.
//
///////////////////////////////////////////////////////////////////////////////
template<class Type> inline Type Min(Type valueA, Type valueB)
{
    return ((valueA < valueB) ? valueA : valueB);
}// Min


///////////////////////////////////////////////////////////////////////////////
//
//      Get the maximum of two values.  Note:  operator < must be defined for
//  the given type.
//
///////////////////////////////////////////////////////////////////////////////
template<class Type> inline Type Max(Type valueA, Type valueB)
{
    return ((valueA < valueB) ? valueB : valueA);
}// Max


///////////////////////////////////////////////////////////////////////////////
//
//      Convert radians to degrees.
//
///////////////////////////////////////////////////////////////////////////////
inline float RadiansToDegrees(float angle)
{
    return angle * 180.f / c_pi;
}// RadiansToDegrees


///////////////////////////////////////////////////////////////////////////////
//
//      Convert degress to radians.
//
///////////////////////////////////////////////////////////////////////////////
inline float DegreesToRadians(float angle)
{
    return angle * c_pi / 180;
}// DegreesToRadians


///////////////////////////////////////////////////////////////////////////////
//
//      Functor to delete an object.
//
///////////////////////////////////////////////////////////////////////////////
/*template<class T> struct FDelete : public std::unary_function<T, void>
{
    void operator ()(T object)
    {
        delete object;
    }// operator ()
};// FDelete
*/

