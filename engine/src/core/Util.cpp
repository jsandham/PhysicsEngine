#include "../../include/core/Util.h"
#include "../../include/core/Types.h"

#include <iostream>
#include <fstream>

using namespace PhysicsEngine;

bool Util::writeToBMP(const std::string &filepath, const std::vector<unsigned char> &data, int width, int height,
                      int numChannels)
{
    if (numChannels != 1 && numChannels != 2 && numChannels != 3 && numChannels != 4)
    {
        return false;
    }

    int size = static_cast<int>(data.size());
    if (size != width * height * numChannels)
    {
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
        return false;
    }

    return true;
}

bool Util::writeToBMP(const std::string &filepath, const std::vector<float> &data, int width, int height,
                      int numChannels)
{
    if (numChannels != 1 && numChannels != 2 && numChannels != 3 && numChannels != 4)
    {
        return false;
    }

    int size = static_cast<int>(data.size());
    if (size != width * height * numChannels)
    {
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
        return false;
    }

    return true;
}

bool Util::writeToPPM(const std::string &filepath, const std::vector<unsigned char> &data, int width, int height)
{
    if (width * height * 3 != data.size())
    {
        return false;
    }

    std::ofstream file;
    file.open(filepath.c_str(), std::ios::out);

    if (file.is_open())
    {
        file << std::to_string(width) << " " << std::to_string(height) << "\n";
        file << "255\n";
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                file << (int)data[3 * width * r + 3 * c + 0] << " ";
                file << (int)data[3 * width * r + 3 * c + 1] << " ";
                file << (int)data[3 * width * r + 3 * c + 2] << "\n";
            }
        }

        file.close();
    }
    else
    {
        std::cout << "Could not open file" << std::endl;
        return false;
    }

    return true;
}





bool Util::isAssetYamlExtension(const std::string &extension)
{
    if (extension == TEXTURE2D_EXT || extension == MESH_EXT || extension == SHADER_EXT || extension == MATERIAL_EXT ||
        extension == SPRITE_EXT || extension == RENDERTEXTURE_EXT || extension == CUBEMAP_EXT)
    {
        return true;
    }

    return false;
}

bool Util::isTextureYamlExtension(const std::string &extension)
{
    return extension == TEXTURE2D_EXT;
}

bool Util::isMaterialYamlExtension(const std::string &extension)
{
    return extension == MATERIAL_EXT;
}

bool Util::isMeshYamlExtension(const std::string &extension)
{
    return extension == MESH_EXT;
}

bool Util::isShaderYamlExtension(const std::string &extension)
{
    return extension == SHADER_EXT;
}

bool Util::isTextureExtension(const std::string &extension)
{
    return (extension == PNG_EXT || extension == JPG_EXT);
}

bool Util::isMeshExtension(const std::string &extension)
{
    return extension == OBJ_EXT;
}

bool Util::isShaderExtension(const std::string &extension)
{
    return (extension == GLSL_EXT || extension == HLSL_EXT);
}