STRINGIFY(
layout (location = 0) out vec3 positionTex;
layout (location = 1) out vec3 normalTex;

in vec3 FragPos;
in vec3 Normal;

void main()
{
    // store the fragment position vector in the first gbuffer texture
    positionTex = FragPos.xyz;
    // also store the per-fragment normals into the gbuffer
    normalTex = normalize(Normal);
}
)