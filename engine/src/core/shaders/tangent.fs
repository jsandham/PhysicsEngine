STRINGIFY(
uniform int wireframe;
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;
void main(void)
{
    // derivations of the fragment position
    vec3 pos_dx = dFdx(FragPos);
    vec3 pos_dy = dFdy(FragPos);
    // derivations of the texture coordinate
    vec2 texC_dx = dFdx(TexCoord);
    vec2 texC_dy = dFdy(TexCoord);
    // tangent vector and binormal vector
    vec3 tangent = texC_dy.y * pos_dx - texC_dx.y * pos_dy;
    tangent = tangent - Normal * dot(tangent, Normal);
    FragColor = vec4(normalize(tangent), 1.0f);
}
)