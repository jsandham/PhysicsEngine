#vertex
#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoords;
out vec2 TexCoords;
void main()
{
	TexCoords = aTexCoords;
	gl_Position = vec4(aPos, 1.0);
}

#fragment
#version 430 core
out vec4 FragColor;
in vec2 TexCoords;
uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
struct Light {
	vec3 Position;
	vec3 Color;
};
const int NR_LIGHTS = 32;
uniform Light lights[NR_LIGHTS];
uniform vec3 viewPos;
void main()
{
	// retrieve data from G-buffer
	vec3 FragPos = texture(gPosition, TexCoords).rgb;
	vec3 Normal = texture(gNormal, TexCoords).rgb;
	vec3 Albedo = texture(gAlbedoSpec, TexCoords).rgb;
	float Specular = texture(gAlbedoSpec, TexCoords).a;
	// then calculate lighting as usual
	vec3 lighting = Albedo * 0.1; // hard-coded ambient component
	vec3 viewDir = normalize(viewPos - FragPos);
	for (int i = 0; i < NR_LIGHTS; ++i)
	{
		// diffuse
		vec3 lightDir = normalize(lights[i].Position - FragPos);
		vec3 diffuse = max(dot(Normal, lightDir), 0.0) * Albedo * lights[i].Color;
		lighting += diffuse;
	}
	FragColor = vec4(lighting, 1.0);
}
