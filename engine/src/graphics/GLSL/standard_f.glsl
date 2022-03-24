#version 430 core
#include "light.glsl"
#include "material.glsl"

uniform Material material;
uniform sampler2D shadowMap[5];

in vec3 FragPos;
in vec3 CameraPos;
in vec3 Normal;
in vec2 TexCoord;
in float ClipSpaceZ;
in vec4 FragPosLightSpace[5];

out vec4 FragColor;

#include "calc_shadow.glsl"
#include "directional.glsl"
#include "spotlight.glsl"
#include "pointlight.glsl"

void main(void)
{
    vec3 viewDir = normalize(CameraPos - FragPos);
    vec4 albedo = vec4(material.colour, 1.0);
    if(material.sampleMainTexture == 1)
    {
        albedo = texture(material.mainTexture, TexCoord);
    }

#if defined (DIRECTIONALLIGHT)
    FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * albedo;
#elif defined(SPOTLIGHT)
    FragColor = vec4(CalcSpotLight(material, Normal, FragPos, viewDir), 1.0f) * albedo;
#elif defined(POINTLIGHT)
    FragColor = vec4(CalcPointLight(material, Normal, FragPos, viewDir), 1.0f) * albedo;
#else
    FragColor = vec4(0.5, 0.5, 0.5, 1.0) * albedo;
#endif
}
