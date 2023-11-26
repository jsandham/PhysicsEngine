// INPUT/OUTPUT structures
struct VS_INPUT
{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
};

// uniforms : external parameters
cbuffer CAMERA_CONSTANT_BUFFER : register(b0)
{
    float3 cameraPos;
    matrix view;
    matrix projection;
    matrix viewProjection;
}

cbuffer VS_CONSTANT_BUFFER2 : register(b1)
{
    matrix model;
}

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    VS_OUTPUT output;
    output.position = mul(viewProjection * model, float4(input.position, 1.0));
    output.normal = input.normal;
    return output;
}