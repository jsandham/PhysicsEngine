#ifndef __MATERIAL_LOADER_H__
#define __MATERIAL_LOADER_H__

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>

#include "Shader.h"
#include "Guid.h"
#include "../json/json.hpp"

namespace PhysicsEngine
{
	typedef struct material_data
	{
		Guid shaderId;
		std::vector<ShaderUniform> uniforms;
	}material_data;

	bool mat_load(const std::string& filepath, material_data& mat)
	{
		std::ifstream file(filepath, std::ios::in);
		std::ostringstream contents;
		if (file.is_open()) {
			contents << file.rdbuf();
			file.close();
		}
		else {
			return false;
		}

		json::JSON jsonMaterial = json::JSON::Load(contents.str());

		json::JSON::JSONWrapper<std::map<std::string, json::JSON>> objects = jsonMaterial.ObjectRange();
		std::map<std::string, json::JSON>::iterator it;

		// really we only need to serialize the uniform name and the corresponding data here...
		std::vector<ShaderUniform> uniforms;
		for (it = objects.begin(); it != objects.end(); it++) {

			if (it->first == "shader") {
				mat.shaderId = Guid(it->second.ToString());

				continue;
			}

			ShaderUniform uniform;

			if (it->second["type"].ToInt() == (int)GL_INT) {
				int value = it->second["data"].ToInt();
				memcpy((void*)uniform.data, &value, sizeof(int));
			}
			else if (it->second["type"].ToInt() == (int)GL_FLOAT) {
				float value = (float)it->second["data"].ToFloat();
				memcpy((void*)uniform.data, &value, sizeof(float));
			}
			else if (it->second["type"].ToInt() == (int)GL_FLOAT_VEC2) {
				glm::vec2 vec = glm::vec2(0.0f);
				vec.x = (float)it->second["data"][0].ToFloat();
				vec.y = (float)it->second["data"][1].ToFloat();
				memcpy((void*)uniform.data, &vec, sizeof(glm::vec2));
			}
			else if (it->second["type"].ToInt() == (int)GL_FLOAT_VEC3) {
				glm::vec3 vec = glm::vec3(0.0f);
				vec.x = (float)it->second["data"][0].ToFloat();
				vec.y = (float)it->second["data"][1].ToFloat();
				vec.z = (float)it->second["data"][2].ToFloat();
				memcpy((void*)uniform.data, &vec, sizeof(glm::vec3));
			}
			else if (it->second["type"].ToInt() == (int)GL_FLOAT_VEC4) {
				glm::vec4 vec = glm::vec4(0.0f);
				vec.x = (float)it->second["data"][0].ToFloat();
				vec.y = (float)it->second["data"][1].ToFloat();
				vec.z = (float)it->second["data"][2].ToFloat();
				vec.w = (float)it->second["data"][3].ToFloat();
				memcpy((void*)uniform.data, &vec, sizeof(glm::vec4));
			}

			if (it->second["type"].ToInt() == (int)GL_SAMPLER_2D) {
				Guid textureId = Guid(it->second["data"].ToString());
				std::string test = textureId.toString();
				memcpy((void*)uniform.data, &textureId, sizeof(Guid));
			}

			std::string name = it->first;
			std::string shortName = it->second["shortName"].ToString();
			std::string blockName = it->second["blockName"].ToString();

			for (size_t i = 0; i < name.length(); i++) {
				uniform.name[i] = name[i];
			}
			for (size_t i = name.length(); i < 32; i++) {
				uniform.name[i] = '\0';
			}

			for (size_t i = 0; i < shortName.length(); i++) {
				uniform.shortName[i] = shortName[i];
			}
			for (size_t i = shortName.length(); i < 32; i++) {
				uniform.shortName[i] = '\0';
			}

			for (size_t i = 0; i < blockName.length(); i++) {
				uniform.blockName[i] = blockName[i];
			}
			for (size_t i = blockName.length(); i < 32; i++) {
				uniform.blockName[i] = '\0';
			}

			uniform.nameLength = it->second["nameLength"].ToInt();
			uniform.size = it->second["size"].ToInt();
			uniform.type = it->second["type"].ToInt();
			uniform.variant = it->second["variant"].ToInt();
			uniform.location = it->second["location"].ToInt();
			uniform.index = it->second["index"].ToInt();

			uniforms.push_back(uniform);
		}

		mat.uniforms = uniforms;

		return true;
	}
}

#endif