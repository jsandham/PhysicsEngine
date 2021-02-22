#include "../../include/core/Util.h"

#include <iostream>

using namespace PhysicsEngine;

bool Util::writeToBMP(const std::string &filepath, std::vector<unsigned char> &data, int width, int height,
                      int numChannels)
{
    if (numChannels != 1 && numChannels != 2 && numChannels != 3 && numChannels != 4)
    {
        std::cout << "TextureLoader: Number of channels must be 1, 2, 3, or 4 where each channel is 8 bits"
                  << std::endl;
        return false;
    }

    if (data.size() != width * height * numChannels)
    {
        std::cout << data.size() << " " << width << " " << height << " " << numChannels << std::endl;
        std::cout << "TextureLoader: Data does not match width, height, and number of channels given" << std::endl;
        return false;
    }

    std::vector<unsigned char> formatedData;
    if (numChannels == 1)
    {
        formatedData.resize(3 * width * height);
        for (int i = 0; i < width * height; i++)
        {
            formatedData[3 * i] = data[i];
            formatedData[3 * i + 1] = data[i];
            formatedData[3 * i + 2] = data[i];
        }
        numChannels = 3;
    }
    else
    {
        formatedData.resize(numChannels * width * height);
        for (int i = 0; i < numChannels * width * height; i++)
        {
            formatedData[i] = data[i];
        }
    }

    BMPHeader header = {};

    header.fileType = 0x4D42;
    header.fileSize = sizeof(BMPHeader) + (unsigned int)formatedData.size();
    header.bitmapOffset = sizeof(BMPHeader);
    header.size = sizeof(BMPHeader) - 14;
    header.width = width;
    header.height = height;
    header.planes = 1;
    header.bitsPerPixel = (unsigned short)(numChannels * 8);
    header.compression = 0;
    header.sizeOfBitmap = (unsigned int)formatedData.size();
    header.horizontalResolution = 0;
    header.verticalResolution = 0;
    header.colorsUsed = 0;
    header.colorsImportant = 0;

    /*FILE *file = fopen(filepath.c_str(), "wb");*/
    FILE *file;
    fopen_s(&file, filepath.c_str(), "wb");
    if (file)
    {
        fwrite(&header, sizeof(BMPHeader), 1, file);
        fwrite(&formatedData[0], formatedData.size(), 1, file);
        fclose(file);
    }
    else
    {
        std::cout << "TextureLoader: Failed to open file " << filepath << " for writing" << std::endl;
        return false;
    }

    std::cout << "TextureLoader: Screen capture successful" << std::endl;

    return true;
}

bool Util::writeToBMP(const std::string &filepath, std::vector<float> &data, int width, int height, int numChannels)
{
    if (numChannels != 1 && numChannels != 2 && numChannels != 3 && numChannels != 4)
    {
        std::cout << "TextureLoader: Number of channels must be 1, 2, 3, or 4 where each channel is 8 bits"
                  << std::endl;
        return false;
    }

    if (data.size() != width * height * numChannels)
    {
        std::cout << data.size() << " " << width << " " << height << " " << numChannels << std::endl;
        std::cout << "TextureLoader: Data does not match width, height, and number of channels given" << std::endl;
        return false;
    }

    std::vector<unsigned char> formatedData;
    if (numChannels == 1)
    {
        formatedData.resize(3 * width * height);
        for (int i = 0; i < width * height; i++)
        {
            formatedData[3 * i] = (unsigned char)(255 * data[i]);
            formatedData[3 * i + 1] = (unsigned char)(255 * data[i]);
            formatedData[3 * i + 2] = (unsigned char)(255 * data[i]);
        }
        numChannels = 3;
    }
    else
    {
        formatedData.resize(numChannels * width * height);
        for (int i = 0; i < numChannels * width * height; i++)
        {
            formatedData[i] = (unsigned char)(255 * data[i]);
        }
    }

    BMPHeader header = {};

    header.fileType = 0x4D42;
    header.fileSize = sizeof(BMPHeader) + (unsigned int)formatedData.size();
    header.bitmapOffset = sizeof(BMPHeader);
    header.size = sizeof(BMPHeader) - 14;
    header.width = width;
    header.height = height;
    header.planes = 1;
    header.bitsPerPixel = (unsigned short)(numChannels * 8);
    header.compression = 0;
    header.sizeOfBitmap = (unsigned int)formatedData.size();
    header.horizontalResolution = 0;
    header.verticalResolution = 0;
    header.colorsUsed = 0;
    header.colorsImportant = 0;

    /*FILE *file = fopen(filepath.c_str(), "wb");*/
    FILE *file;
    fopen_s(&file, filepath.c_str(), "wb");
    if (file)
    {
        fwrite(&header, sizeof(BMPHeader), 1, file);
        fwrite(&formatedData[0], formatedData.size(), 1, file);
        fclose(file);
    }
    else
    {
        std::cout << "TextureLoader: Failed to open file " << filepath << " for writing" << std::endl;
        return false;
    }

    std::cout << "TextureLoader: Screen capture successful" << std::endl;

    return true;
}