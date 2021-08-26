out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
uniform vec3 lightPos;
uniform vec4 color;
void main()
{
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(abs(dot(norm, lightDir)), 0.1);
    vec4 diffuse = vec4(diff, diff, diff, 1.0);
    FragColor = diffuse * color;
}
