#include "../../include/core/GLM.h"

using namespace YAML;

template <> std::string YAML::getValue<std::string>(const Node &node, const std::string &key)
{
    if (node[key])
    {
        std::string temp = node[key].as<std::string>();
        return temp.compare("null") == 0 ? "" : temp;
    }

    return "";
}