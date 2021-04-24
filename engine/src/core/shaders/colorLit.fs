STRINGIFY(
uniform vec3 lightDirection;
uniform vec3 color;
uniform int wireframe;
in vec3 FragPos;
in vec3 CameraPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

vec3 CalcDirLight(vec3 normal, vec3 viewDir);

void main(void)
{
    vec3 viewDir = normalize(CameraPos - FragPos);
    FragColor = vec4(CalcDirLight(Normal, viewDir) * color, 1.0f);
}

vec3 CalcDirLight(vec3 normal, vec3 viewDir)
{
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightDirection);

    vec3 reflectDir = reflect(-lightDir, norm);

    float diffuseStrength = max(dot(norm, lightDir), 0.0);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), 1.0f);

    vec3 ambient = vec3(0.7, 0.7, 0.7);
    vec3 diffuse = vec3(1.0, 1.0, 1.0) * diffuseStrength;
    vec3 specular = vec3(0.7, 0.7, 0.7) * specularStrength;

    return (ambient + diffuse + specular);
}
)