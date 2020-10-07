#ifndef __UTIL_H__
#define __UTIL_H__

#include <string>
#include <vector>

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct BMPHeader
{
    unsigned short fileType;
    unsigned int fileSize;
    unsigned short reserved1;
    unsigned short reserved2;
    unsigned int bitmapOffset;
    unsigned int size;
    int width;
    int height;
    unsigned short planes;
    unsigned short bitsPerPixel;
    unsigned int compression;
    unsigned int sizeOfBitmap;
    int horizontalResolution;
    int verticalResolution;
    unsigned int colorsUsed;
    unsigned int colorsImportant;
};
#pragma pack(pop)

template <typename T, typename U, typename V> struct triple
{
    T first;
    U second;
    V third;
};

template <typename T, typename U, typename V> triple<T, U, V> make_triple(T first, U second, V third)
{
    triple<T, U, V> triple;
    triple.first = first;
    triple.second = second;
    triple.third = third;

    return triple;
}

class Util
{
  public:
    static bool writeToBMP(const std::string &filepath, std::vector<unsigned char> &data, int width, int height,
                           int numChannels);
    static bool writeToBMP(const std::string &filepath, std::vector<float> &data, int width, int height,
                           int numChannels);
};
} // namespace PhysicsEngine

#endif