[Begin_ResourceLayout]

	[Directives:DiffuseIrradiance DI_OFF DI]
	[Directives:SpecularRadiance SR_OFF SR]
	[Directives:Albedo DIFF_OFF DIFF]
	[Directives:Normal NORMAL_OFF NORMAL]
	[Directives:Normal PARALLAX_OFF PARALLAX]
	[Directives:MetallicRoughnessAOTexture MRA_OFF MRA]
	[Directives:ShadowSupported SHADOW_SUPPORTED_OFF SHADOW_SUPPORTED]
	[Directives:ColorSpace GAMMA_COLORSPACE_OFF GAMMA_COLORSPACE]
	
	cbuffer PerDrawCall : register(b0)
	{
		float4x4 WorldViewProj	: packoffset(c0);	[WorldViewProjection]
		float4x4 World			: packoffset(c4); 	[World]
	};

	cbuffer PerCamera : register(b1)
	{
		float3 CameraPosition		: packoffset(c0.x); [CameraPosition]
		uint IblMaxMipLevel			: packoffset(c0.w); [IBLMipMapLevel]
	}
	
	cbuffer Parameters : register(b2)
	{
		float3 SunDirection			: packoffset(c0.x); [SunDirection]
		float3 SunColor				: packoffset(c1.x); [SunColor]
		float Metallic				: packoffset(c2.x); [Default(0)]
		float Roughness				: packoffset(c2.y); [Default(0)]
		float Reflectance			: packoffset(c2.z); [Default(0.3)]
		float IrradiPerp 			: packoffset(c2.w); [Default(3)]
		float3 BaseColor			: packoffset(c3.x); [Default(1,1,1)]
	};

	Texture2D BaseTexture								: register(t0);
	Texture2D MetallicRoughnessAOTexture     			: register(t1);
	Texture2D NormalHeightTexture						: register(t2);
	TextureCube IBLRadianceTexture						: register(t3); [IBLRadiance]
	TextureCube IBLIrradianceTexture					: register(t4); [IBLIrradiance]
	Texture2DArray DirectionalShadowMap					: register(t5); [DirectionalShadowMap]
	
	SamplerState TextureSampler							: register(s0);
	SamplerState IBLRadianceSampler						: register(s1);
	SamplerState IBLIrradianceSampler					: register(s2);
	SamplerComparisonState DirectionalShadowMapSampler	: register(s3);	
	
[End_ResourceLayout]

[Begin_Pass:Default]
	[Profile 10_0]
	[Entrypoints VS=VS PS=PS]

	#define PI 3.14159265359f
	
	struct VS_IN
	{
		float4 position : POSITION;
		float3 normal	: NORMAL;
		
		#if NORMAL
		float4 tangent	: TANGENT;
		#endif
		
		#if DIFF || NORMAL || MRA
		float2 texCoord : TEXCOORD;
		#endif
	};

	struct PS_IN
	{
		float4 position 	: SV_POSITION;
		float3 normal		: NORMAL0;
		
		#if NORMAL
		float3 tangent		: TANGENT0;
		float3 bitangent	: BINORMAL0;
		#endif
		
		#if DIFF || NORMAL || MRA
		float2 texCoord 	: TEXCOORD0;
		#endif
		
		float3 positionWS 	: TEXCOORD1;
	};

	struct Surface
	{
		half3 albedo;
		half AO;
		half3 position;
		half NdotV;
		half3 normal;
		half reflectance;
		half3 viewVector;
		half metallic;
		half3 reflectVector;
		half roughness;
		
		
		inline void Create( in half3 color,
							in half3 P,
							in half3 N,	
							in half3 viewPos,
							in half smetallic,
							in half sroughness,
							in half sAO,
							in half sreflectance)
		{
			albedo = color;
			position = P;
			normal = N;
			viewVector = normalize(viewPos - P);
			reflectVector = 0;
			NdotV = saturate(dot(N, viewPos));
			AO = sAO;
			metallic = smetallic;
			roughness = sroughness * sroughness;
			reflectance = sreflectance;
			
			#if SR
			reflectVector = normalize(reflect(-viewVector, N));
			#endif
		}
	};
	
	half3 fresnelSchlick(half cosTheta, half3 F0)
	{
  		return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
	} 
	
	struct SurfaceToLight
	{
		half3 lightVector;
		half NdotL;
		half3 halfVector;
		half NdotH;
		half3 fresnelTerm;
		half VdotH;
		half irradiance;
		
		inline void Create(in Surface surface, in half3 lightDir)
		{
			lightVector = lightDir;
			halfVector = normalize(lightDir + surface.viewVector);
			NdotL = saturate(dot(lightVector, surface.normal));
			NdotH = saturate(dot(surface.normal, halfVector));
			VdotH = saturate(dot(surface.viewVector, halfVector));
			
			float3 f0 = 0.16 * (surface.reflectance * surface.reflectance);
			f0 = lerp(f0, surface.albedo, surface.metallic);
			fresnelTerm = fresnelSchlick(VdotH, f0);
			irradiance = max(dot(lightVector, surface.normal), 0.0) * IrradiPerp;
		}
	};

	PS_IN VS(VS_IN input)
	{
		PS_IN output = (PS_IN)0;

		output.position = mul(input.position, WorldViewProj);
		
		output.positionWS = mul(input.position, World).xyz;
		output.normal = mul(float4(input.normal, 0), World).xyz;
		
		#if NORMAL
		output.tangent = mul(input.tangent, World).xyz;
		output.bitangent = cross(output.normal, output.tangent) * input.tangent.w;
		#endif
		
		#if DIFF || NORMAL || MRA
		output.texCoord = input.texCoord;
		#endif
		
		return output;
	}

	half3 GammaToLinear(in half3 color)
	{
		return pow(abs(color), 2.2);
	}
	
	half3 LinearToGamma(const half3 color)
	{
		return pow(color, 1 / 2.2);
	}
	
	half D_GGX(half NoH, half roughness)
	{
		half alpha = roughness * roughness;
		half alpha2 = alpha * alpha;
		half NoH2 = NoH * NoH;
		half b = (NoH2 * (alpha2 - 1.0) + 1.0);
		return alpha2 / (PI * b * b);
	}
	
	half G1_GGX_Schlick(half NdotV, half roughness)
	{
		//float r = roughness; // original
		half r = 0.5 + 0.5 * roughness; // Disney remapping
		half k = (r * r) / 2.0;
		half denom = NdotV * (1.0 - k) + k;
		return NdotV / denom;
	}
	
	half G_Smith(half NdotV, half NdotL, half roughness) 
	{
		half g1_l = G1_GGX_Schlick(NdotL, roughness);
		half g1_v = G1_GGX_Schlick(NdotV, roughness);
		return g1_l * g1_v;
	}
	
	half3 BRDFSpecular(in Surface surface, in SurfaceToLight surface2light)
	{
		half3 radiacenSpecular = 0;
		
		#if SR
		float lod = IblMaxMipLevel * surface.roughness;
		radiacenSpecular = IBLRadianceTexture.SampleLevel(IBLRadianceSampler, surface.reflectVector, lod).rgb;
		#endif
		
		half3 F = surface2light.fresnelTerm;
		half D = D_GGX(surface2light.NdotH, surface.roughness);
		half G = G_Smith(surface.NdotV, surface2light.NdotL, surface.roughness);
		
		half3 specular = (D * G * F) / max(4.0 * surface.NdotV * surface2light.NdotL, 0.001);
		
		return specular * surface2light.irradiance + radiacenSpecular * surface.metallic;
	}
	
	half3 BRDFDiffuse(in Surface surface, in SurfaceToLight surface2light)
	{
		half3 ambient = 0;
		
		#if DI
		half3 diffuseIrradiance = IBLIrradianceTexture.Sample(IBLIrradianceSampler, surface.normal).rgb;
		ambient = surface.albedo * surface.AO * diffuseIrradiance;
		#endif
		
		half3 rhoD = 1.0 - surface2light.fresnelTerm; // if not specular, use as diffuse
		rhoD *= 1.0 - surface.metallic; // no diffuse for metals
		
		half3 diffuse = rhoD * surface.albedo / PI;
		
		return ambient + diffuse * surface2light.irradiance;
	}

	float4 PS(PS_IN input) : SV_Target
	{
		half3 base = BaseColor;
		
		#if DIFF
		base = GammaToLinear(BaseTexture.Sample(TextureSampler, input.texCoord).rgb);
		#endif
		
		half3 normal = normalize(input.normal);
		
		#if NORMAL
		half3 normalTex = NormalHeightTexture.Sample(TextureSampler, input.texCoord).rgb * 2 - 1;
		float3x3 tangentToWorld = float3x3(normalize(input.tangent), normalize(input.bitangent), normalize(input.normal));
		normal = normalize(mul(normalTex, tangentToWorld));
		#endif
		
		half metallic = Metallic;
		half roughness = Roughness;
		half AO = 1;

		#if MRA
		half3 MetallicRoughnessAO = MetallicRoughnessAOTexture.Sample(TextureSampler, input.texCoord).xyz;
		
		metallic = MetallicRoughnessAO.x;
		roughness = MetallicRoughnessAO.y;
		AO = MetallicRoughnessAO.z;
		#endif
		
		Surface surface;
		surface.Create(base,input.positionWS, normal, CameraPosition, metallic, roughness, AO, Reflectance);
		
		SurfaceToLight surface2light;
		surface2light.Create(surface, SunDirection);
		
		half3 diffuse = BRDFDiffuse(surface, surface2light);
		half3 specular = BRDFSpecular(surface, surface2light);
		
		half3 color = (diffuse + specular) * SunColor;
		
		#if GAMMA_COLORSPACE
		color = LinearToGamma(color);
		#endif
		
		return float4(color, 1.0);
	}

[End_Pass]