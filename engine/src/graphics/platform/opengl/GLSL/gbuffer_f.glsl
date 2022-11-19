#version 430 core
#include "material.glsl"
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;
in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;

uniform Material material;
void main()
{
  // store the fragment position vector in the first gbuffer texture
  gPosition = FragPos;
  // also store the per-fragment normals into the gbuffer
  gNormal = normalize(Normal);
  // and the diffuse per-fragment color
  gAlbedoSpec.rgb = texture(material.mainTexture, TexCoords).rgb;
  // store specular intensity in gAlbedoSpec's alpha component
  gAlbedoSpec.a = 1.0;//texture(texture_specular1, TexCoords).r;
}