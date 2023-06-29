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
matrix worldViewProjection;
vector aVec;

cbuffer Test1 
{
    matrix wvp;
    //vector cameraPos;
    float alpha;
};

cbuffer Test2
{
    matrix view;
    // vector cameraPos;
    // float alpha;
};

cbuffer Test3
{
    matrix model;
    // vector cameraPos;
    // float alpha;
};

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    VS_OUTPUT output;
    output.position = mul(worldViewProjection, float4(input.position, 1.0));
    output.normal = input.normal;

    float4 temp1 = mul(wvp, float4(input.position, 1.0));
    float4 temp2 = mul(view, float4(input.position, 1.0));
    float4 temp3 = mul(model, float4(input.position, 1.0));

    output.position.x = alpha * temp1.x;
    output.position.y = alpha * temp2.y;
    output.position.z = alpha * temp3.z;

    return output;
}