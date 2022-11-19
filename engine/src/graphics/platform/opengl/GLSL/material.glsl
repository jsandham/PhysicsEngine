struct Material
{
    float shininess;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 colour;
    sampler2D mainTexture;
    sampler2D normalMap;
    sampler2D specularMap;

    int sampleMainTexture;
    int sampleNormalMap;
    int sampleSpecularMap;
};