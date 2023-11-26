#vertex
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
//cbuffer CAMERA_CONSTANT_BUFFER : register(b0)
//{
//    float3 cameraPos;
//    matrix view;
//    matrix projection;
//    matrix viewProjection;
//}

cbuffer VS_CONSTANT_BUFFER2 : register(b0)
{
    matrix projection;
    matrix view;
    matrix model;
}

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    float4x4 modelViewProj = mul(model, mul(view, projection));
    VS_OUTPUT output;
    output.position = mul(modelViewProj, float4(input.position, 1.0));
    output.normal = input.normal;
    return output;
}

#fragment
struct PS_INPUT
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
};

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return float4(0.0f, 0.0f, 1.0f, 1.0f);
    //return float4(input.normal, 1.0);
}