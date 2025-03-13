# AWS Bedrock Capabilities for Online Processing and Batch Inference

This document provides a comprehensive overview of AWS Bedrock capabilities for both online processing and batch inference, including supported models, region availability, inference profiles, input/output formats, and implementation considerations.

## Table of Contents

- [Online Processing](#online-processing)
  - [Supported Models and Regions](#supported-models-and-regions)
  - [Input/Output Formats](#inputoutput-formats)
  - [Inference Profiles](#inference-profiles)
  - [API Reference](#api-reference)
  - [Implementation Considerations](#implementation-considerations)
- [Batch Inference](#batch-inference)
  - [Supported Models and Regions](#supported-models-and-regions-1)
  - [Input/Output Formats](#inputoutput-formats-1)
  - [Batch Job Creation and Management](#batch-job-creation-and-management)
  - [Implementation Considerations](#implementation-considerations-1)
- [Cross-Region Considerations](#cross-region-considerations)
- [Authentication and Permissions](#authentication-and-permissions)
- [Error Handling](#error-handling)
- [Monitoring and Logging](#monitoring-and-logging)

## Online Processing

Online processing in AWS Bedrock refers to real-time, synchronous API calls to foundation models for generating responses to individual prompts.

### Supported Models and Regions

AWS Bedrock is available in multiple regions, with each region supporting different models. The primary regions include:

| Region Name | Region Code |
|-------------|-------------|
| US East (N. Virginia) | us-east-1 |
| US East (Ohio) | us-east-2 |
| US West (Oregon) | us-west-2 |
| AWS GovCloud (US-West) | us-gov-west-1 |
| Asia Pacific (Tokyo) | ap-northeast-1 |
| Asia Pacific (Seoul) | ap-northeast-2 |
| Asia Pacific (Mumbai) | ap-south-1 |
| Asia Pacific (Singapore) | ap-southeast-1 |
| Asia Pacific (Sydney) | ap-southeast-2 |
| Canada (Central) | ca-central-1 |
| Europe (Frankfurt) | eu-central-1 |
| Europe (Zurich) | eu-central-2 |
| Europe (Ireland) | eu-west-1 |
| Europe (London) | eu-west-2 |
| Europe (Paris) | eu-west-3 |
| South America (São Paulo) | sa-east-1 |

#### Detailed Model Availability by Region

Below is a comprehensive breakdown of region availability for each model, distinguishing between online processing and batch inference capabilities.

##### Amazon Models

###### Amazon Titan Text Models

**Amazon Titan Text G1 - Express**
- **Online processing**: us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1
- **Batch inference**: us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, eu-central-1

**Amazon Titan Text G1 - Lite**
- **Online processing**: us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1
- **Batch inference**: us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, eu-central-1

**Amazon Titan Text G1 - Premier**
- **Online processing**: us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1
- **Batch inference**: us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, eu-central-1

###### Amazon Titan Embeddings Models

**Amazon Titan Multimodal Embeddings G1**
- **Online processing**: us-east-1, us-west-2, ap-northeast-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1
- **Batch inference**: us-east-1, us-west-2, ap-northeast-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1

**Amazon Titan Text Embeddings V2**
- **Online processing**: us-east-1, us-west-2, ap-northeast-2, ca-central-1, eu-central-1, eu-west-2, sa-east-1
- **Batch inference**: us-east-1, us-west-2, ap-northeast-2, ca-central-1, eu-central-1, eu-west-2, sa-east-1

###### Amazon Nova Models

**Amazon Nova Lite, Micro, and Pro**
- **Online processing**: us-east-1
- **Batch inference**: us-east-1

##### Anthropic Models

###### Claude 3 Family

**Claude 3 Opus**
- **Online processing**: us-east-1, us-west-2
- **Batch inference**: us-west-2 only

**Claude 3 Sonnet**
- **Online processing**: us-east-1, us-west-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1
- **Batch inference**: us-east-1, us-west-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1

**Claude 3 Haiku**
- **Online processing**: us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1
- **Batch inference**: us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1

###### Claude 3.5 Family

**Claude 3.5 Sonnet**
- **Online processing**: us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-southeast-1, eu-central-1
- **Batch inference**: us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-southeast-1, eu-central-1

**Claude 3.5 Sonnet v2**
- **Online processing**: us-west-2
- **Batch inference**: us-west-2 only

**Claude 3.5 Haiku**
- **Online processing**: us-west-2
- **Batch inference**: us-west-2 only

**Claude 3.7 Sonnet**
- **Online processing**: us-east-1, us-east-2, us-west-2
- **Batch inference**: us-east-1, us-east-2, us-west-2

##### AI21 Labs Models

**Jurassic-2 Models (Mid, Ultra, J2 Jumbo Instruct)**
- **Online processing**: us-east-1 only
- **Batch inference**: us-east-1 only

**Jamba Models (1.5 Large, 1.5 Mini, Jamba-Instruct)**
- **Online processing**: Primarily in us-east-1, us-west-2
- **Batch inference**: us-east-1, us-west-2

##### Meta Llama Models

**Llama 3.1 405B Instruct**
- **Online processing**: us-west-2
- **Batch inference**: us-west-2 only

**Llama 3.1 70B Instruct**
- **Online processing**: us-west-2 and other select regions
- **Batch inference**: us-west-2

**Llama 3.1 8B Instruct and other Llama models**
- **Online processing**: us-east-1, us-east-2, us-west-2, and other select regions
- **Batch inference**: us-east-1, us-west-2

##### Mistral AI Models

**Mistral Large (24.02/24.07) and Mistral Small (24.02)**
- **Online processing**: Multiple regions including us-east-1, us-west-2, eu-west-1
- **Batch inference**: Primarily us-east-1, us-west-2

##### Cohere Models

**Cohere Command, Command R, Command R+**
- **Online processing**: Multiple regions including us-east-1, us-west-2, eu-central-1
- **Batch inference**: us-east-1, us-west-2, select other regions

**Cohere Embed Models**
- **Online processing**: Multiple regions
- **Batch inference**: Available for specific embedding generation tasks in select regions

##### Stability AI Models

**Stability.ai Diffusion Models**
- **Online processing**: Limited regions, primarily us-east-1, us-west-2
- **Batch inference**: Not widely supported for batch inference

#### Model Availability by Provider and Region (Summary)

**Amazon Titan Models**:
- Amazon Titan Text G1 - Express: Available in most regions
- Amazon Titan Text G1 - Lite: Available in most regions
- Amazon Titan Text G1 - Premier: Available in most regions
- Amazon Titan Text Embeddings: Available in most regions
- Amazon Titan Multimodal Embeddings: Available in most regions
- Amazon Titan Image Generator: Available in most regions

**Anthropic Claude Models**:
- Claude 3 Opus: Limited regions (us-east-1, us-west-2)
- Claude 3 Sonnet: Available in most regions, but not in some EU regions
- Claude 3 Haiku: Available in most regions
- Claude 3.5 Sonnet: Available in most regions
- Claude 3.5 Haiku: Available in most regions
- Claude 3.7 Sonnet: Available in most regions
- Claude 2.1: Available in most regions
- Claude 2.0: Available in most regions
- Claude Instant: Available in most regions

**AI21 Labs Models**:
- Jurassic-2 Mid: us-east-1 only
- Jurassic-2 Ultra: us-east-1 only
- J2 Jumbo Instruct: us-east-1 only
- Jamba 1.5 Large: Available in more regions than older Jurassic models
- Jamba 1.5 Mini: Available in more regions than older Jurassic models
- Jamba-Instruct: Available in more regions than older Jurassic models

**Cohere Models**:
- Cohere Command: Multiple regions
- Cohere Command R: Multiple regions
- Cohere Command R+: Multiple regions
- Cohere Embed (English): Multiple regions
- Cohere Embed (Multilingual): Multiple regions

**Meta Llama Models**:
- Llama 2 13B: Multiple regions
- Llama 2 70B: Multiple regions
- Llama 3 8B: Multiple regions
- Llama 3 70B: Multiple regions
- Llama 3.1 8B: Multiple regions
- Llama 3.1 70B: Multiple regions
- Llama 3.1 405B: Limited regions
- Llama 3.2 11B: Multiple regions
- Llama 3.2 90B: Multiple regions
- Llama 3.3 70B: Multiple regions

**Mistral AI Models**:
- Mistral Small (24.02): Multiple regions
- Mistral Large (24.02): Multiple regions
- Mistral Large (24.07): Multiple regions

**Stability AI Models**:
- Stability.ai Diffusion 0.8: Limited regions
- Stability.ai Diffusion 1.0: Limited regions
- Stable Image Core: Limited regions
- Stable Image Ultra: Limited regions
- Stability.ai Stable Diffusion 3: Limited regions

### Input/Output Formats

AWS Bedrock supports different input/output formats based on the model being used. Here are the input/output formats for major model families:

#### Amazon Titan Text Models

**Model IDs**: 
- `amazon.titan-text-express-v1` (Titan Text G1 - Express)
- `amazon.titan-text-lite-v1` (Titan Text G1 - Lite)
- `amazon.titan-text-premier-v1` (Titan Text G1 - Premier)

**Request Format:**
```json
{
    "inputText": "string",
    "textGenerationConfig": {
        "temperature": float,
        "topP": float,
        "maxTokenCount": int,
        "stopSequences": [string]
    }
}
```

**Response Format:**
```json
{
    "inputTextTokenCount": int,
    "results": [{
        "tokenCount": int,
        "outputText": string,
        "completionReason": string
    }]
}
```

#### Amazon Titan Embeddings Models

**Model IDs**:
- `amazon.titan-embed-text-v1` (Titan Embeddings G1 - Text)
- `amazon.titan-embed-text-v2` (Titan Text Embeddings V2)
- `amazon.titan-embed-image-v1` (Titan Multimodal Embeddings G1)

**Request Format (Text Embeddings):**
```json
{
    "inputText": "string"
}
```

**Response Format (Text Embeddings):**
```json
{
    "embedding": [float],
    "inputTextTokenCount": int
}
```

**Request Format (Multimodal Embeddings):**
```json
{
    "inputText": "string",
    "inputImage": {
        "base64": "string"
    }
}
```

**Response Format (Multimodal Embeddings):**
```json
{
    "embedding": [float],
    "inputTextTokenCount": int
}
```

#### Amazon Nova Models

**Model IDs**:
- `amazon.nova-lite-v1` (Nova Lite)
- `amazon.nova-micro-v1` (Nova Micro)
- `amazon.nova-pro-v1` (Nova Pro)

**Request Format:**
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": "string"
                },
                {
                    "video": {
                        "format": "mp4",
                        "source": {
                            "bytes": {
                                "base64": "string"
                            }
                        }
                    }
                },
                {
                    "image": {
                        "format": "jpeg|png",
                        "source": {
                            "bytes": {
                                "base64": "string"
                            }
                        }
                    }
                }
            ]
        }
    ],
    "maxTokens": int,
    "temperature": float,
    "topP": float
}
```

**Response Format:**
```json
{
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "text": "string"
                }
            ]
        }
    },
    "usage": {
        "inputTokens": int,
        "outputTokens": int,
        "totalTokens": int
    }
}
```

#### Anthropic Claude Models

**Model IDs**:
- `anthropic.claude-3-opus-20240229-v1:0` (Claude 3 Opus)
- `anthropic.claude-3-sonnet-20240229-v1:0` (Claude 3 Sonnet)
- `anthropic.claude-3-haiku-20240307-v1:0` (Claude 3 Haiku)
- `anthropic.claude-3-5-sonnet-20240620-v1:0` (Claude 3.5 Sonnet)
- `anthropic.claude-3-5-sonnet-20241022-v2:0` (Claude 3.5 Sonnet v2)
- `anthropic.claude-3-5-haiku-20241022-v1:0` (Claude 3.5 Haiku)
- `anthropic.claude-3-7-sonnet-20250219-v1:0` (Claude 3.7 Sonnet)
- `anthropic.claude-v2:1` (Claude 2.1)
- `anthropic.claude-v2` (Claude 2.0)
- `anthropic.claude-instant-v1` (Claude Instant)

**Messages API Request Format:**
```json
{
    "messages": [
        {
            "role": "user|assistant",
            "content": [
                {
                    "type": "text",
                    "text": "string"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg|image/png",
                        "data": "string"
                    }
                }
            ]
        }
    ],
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": int,
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "stop_sequences": [string]
}
```

**Response Format:**
```json
{
    "id": "string",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "string"
        }
    ],
    "model": "string",
    "stop_reason": "string",
    "stop_sequence": "string",
    "usage": {
        "input_tokens": int,
        "output_tokens": int
    }
}
```

#### AI21 Labs Models

**Model IDs**:
- `ai21.j2-mid-v1` (Jurassic-2 Mid)
- `ai21.j2-ultra-v1` (Jurassic-2 Ultra)
- `ai21.j2-jumbo-instruct` (J2 Jumbo Instruct)
- `ai21.jamba-instruct-v1:0` (Jamba-Instruct)

**Request Format (Jurassic-2):**
```json
{
    "prompt": "string",
    "maxTokens": int,
    "temperature": float,
    "topP": float,
    "stopSequences": [string],
    "countPenalty": {
        "scale": float
    },
    "presencePenalty": {
        "scale": float
    },
    "frequencyPenalty": {
        "scale": float
    }
}
```

**Response Format (Jurassic-2):**
```json
{
    "id": "string",
    "completions": [
        {
            "data": {
                "text": "string",
                "tokens": [
                    {
                        "generatedToken": {
                            "token": "string",
                            "logprob": float,
                            "raw_logprob": float
                        },
                        "topTokens": [
                            {
                                "token": "string",
                                "logprob": float,
                                "raw_logprob": float
                            }
                        ],
                        "textRange": {
                            "start": int,
                            "end": int
                        }
                    }
                ]
            },
            "finishReason": {
                "reason": "string"
            }
        }
    ],
    "prompt": {
        "text": "string",
        "tokens": [
            {
                "generatedToken": {
                    "token": "string",
                    "logprob": float,
                    "raw_logprob": float
                },
                "textRange": {
                    "start": int,
                    "end": int
                }
            }
        ]
    }
}
```

**Request Format (Jamba):**
```json
{
    "messages": [
        {
            "role": "user",
            "content": "string"
        }
    ],
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "max_tokens": int,
    "stream": boolean
}
```

**Response Format (Jamba):**
```json
{
    "id": "string",
    "created_at": "string",
    "message": {
        "role": "assistant",
        "content": "string"
    },
    "model": "string",
    "prompt_tokens": int,
    "completion_tokens": int,
    "stop_reason": "string",
    "stop_sequence": null
}
```

#### Meta Llama Models

**Model IDs**:
- `meta.llama3-1-8b-instruct-v1:0` (Llama 3.1 8B Instruct)
- `meta.llama3-1-70b-instruct-v1:0` (Llama 3.1 70B Instruct)
- `meta.llama3-1-405b-instruct-v1:0` (Llama 3.1 405B Instruct)
- `meta.llama3-2-1b-instruct-v1:0` (Llama 3.2 1B Instruct)
- `meta.llama3-2-3b-instruct-v1:0` (Llama 3.2 3B Instruct)
- `meta.llama3-2-11b-instruct-v1:0` (Llama 3.2 11B Instruct)
- `meta.llama3-2-90b-instruct-v1:0` (Llama 3.2 90B Instruct)
- `meta.llama3-3-70b-instruct-v1:0` (Llama 3.3 70B Instruct)

**Request Format:**
```json
{
    "messages": [
        {
            "role": "user|assistant|system",
            "content": "string"
        }
    ],
    "temperature": float,
    "top_p": float,
    "max_gen_len": int
}
```

**Response Format:**
```json
{
    "generation": "string",
    "prompt_token_count": int,
    "generation_token_count": int,
    "stop_reason": "string"
}
```

#### Mistral AI Models

**Model IDs**:
- `mistral.mistral-small-2402-v1:0` (Mistral Small 24.02)
- `mistral.mistral-large-2402-v1:0` (Mistral Large 24.02)
- `mistral.mistral-large-2407-v1:0` (Mistral Large 24.07)

**Request Format:**
```json
{
    "messages": [
        {
            "role": "user",
            "content": "string"
        }
    ],
    "temperature": float,
    "top_p": float,
    "max_tokens": int
}
```

**Response Format:**
```json
{
    "id": "string",
    "object": "chat.completion",
    "created": int,
    "model": "string",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "string"
            },
            "finish_reason": "string"
        }
    ],
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int
    }
}
```

#### Cohere Models

**Model IDs**:
- `cohere.command-text-v14` (Command)
- `cohere.command-r-v1:0` (Command R)
- `cohere.command-r-plus-v1:0` (Command R+)
- `cohere.embed-english-v3` (Embed English)
- `cohere.embed-multilingual-v3` (Embed Multilingual)

**Request Format (Command):**
```json
{
    "prompt": "string",
    "temperature": float,
    "p": float,
    "k": int,
    "max_tokens": int,
    "stop_sequences": [string],
    "return_likelihoods": "string",
    "stream": boolean,
    "num_generations": int
}
```

**Response Format (Command):**
```json
{
    "generations": [
        {
            "id": "string",
            "text": "string",
            "likelihood": float,
            "token_likelihoods": [
                {
                    "token": "string",
                    "likelihood": float
                }
            ]
        }
    ],
    "id": "string",
    "prompt": "string"
}
```

**Request Format (Command R/R+):**
```json
{
    "message": "string",
    "chat_history": [
        {
            "role": "USER|CHATBOT",
            "message": "string"
        }
    ],
    "temperature": float,
    "p": float,
    "k": int,
    "max_tokens": int,
    "stream": boolean,
    "stop_sequences": [string],
    "preamble": "string"
}
```

**Response Format (Command R/R+):**
```json
{
    "generation_id": "string",
    "text": "string",
    "finish_reason": "string",
    "token_count": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int
    },
    "meta": {
        "api_version": {
            "version": "string"
        }
    }
}
```

**Request Format (Embed):**
```json
{
    "texts": [string],
    "input_type": "string",
    "truncate": "string"
}
```

**Response Format (Embed):**
```json
{
    "id": "string",
    "texts": [string],
    "embeddings": [[float]],
    "meta": {
        "dimensions": int,
        "truncated": [boolean]
    }
}
```

#### Stability AI Models

**Model IDs**:
- `stability.sd3-large-v1:0` (SD3 Large)
- `stability.stable-diffusion-xl-v1` (SDXL 1.0)
- `stability.stable-image-core-v1:0` (Stable Image Core)
- `stability.stable-image-ultra-v1:0` (Stable Image Ultra)

**Request Format (Text-to-Image):**
```json
{
    "text_prompts": [
        {
            "text": "string",
            "weight": float
        }
    ],
    "cfg_scale": float,
    "seed": int,
    "steps": int,
    "style_preset": "string",
    "clip_guidance_preset": "string"
}
```

**Response Format (Text-to-Image):**
Binary image data (with appropriate content-type headers) or base64-encoded image.

**Request Format (Image-to-Image):**
```json
{
    "text_prompts": [
        {
            "text": "string",
            "weight": float
        }
    ],
    "init_image": "string", // base64-encoded image
    "image_strength": float,
    "cfg_scale": float,
    "clip_guidance_preset": "string",
    "seed": int,
    "steps": int
}
```

**Response Format (Image-to-Image):**
Binary image data (with appropriate content-type headers) or base64-encoded image.

### Inference Profiles

AWS Bedrock provides inference profiles that allow users to:

1. **Track usage metrics**: Monitor model invocation with CloudWatch logs
2. **Track costs with tags**: Attach tags to inference profiles for cost attribution
3. **Implement cross-region inference**: Distribute model invocation across regions for higher throughput

#### Types of Inference Profiles

1. **Cross-region (system-defined) inference profiles**: Predefined profiles that include multiple regions
2. **Application inference profiles**: User-created profiles for tracking costs and usage, can route to one or multiple regions

#### System-Defined Cross-Region Inference Profiles

AWS provides predefined inference profiles that route requests across multiple regions automatically. These profiles are immutable and have specific source and destination regions.

**Key Terminology:**
- **Source Region**: The region from which you make the API request specifying the inference profile
- **Destination Region**: A region to which AWS Bedrock can route the request from your source region

##### Available Cross-Region Inference Profiles

| Inference Profile | Profile ID | Source Regions | Destination Regions |
|-------------------|------------|----------------|---------------------|
| US Nova Lite | `us.amazon.nova-lite-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Nova Micro | `us.amazon.nova-micro-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Nova Pro | `us.amazon.nova-pro-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Anthropic Claude 3 Haiku | `us.anthropic.claude-3-haiku-20240307-v1:0` | us-east-1, us-east-2, us-west-2 | Varies by source: <br>- From us-east-1: us-east-1, us-west-2<br>- From us-east-2: us-east-1, us-east-2, us-west-2<br>- From us-west-2: us-east-1, us-west-2 |
| US Anthropic Claude 3 Opus | `us.anthropic.claude-3-opus-20240229-v1:0` | us-east-1, us-west-2 | us-east-1, us-west-2 |
| US Anthropic Claude 3 Sonnet | `us.anthropic.claude-3-sonnet-20240229-v1:0` | us-east-1, us-west-2 | us-east-1, us-west-2 |
| US Anthropic Claude 3.5 Sonnet | `us.anthropic.claude-3-5-sonnet-20240620-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Anthropic Claude 3.5 Haiku | `us.anthropic.claude-3-5-haiku-20241022-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Anthropic Claude 3.5 Sonnet v2 | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Anthropic Claude 3.7 Sonnet | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Meta Llama 3.1 8B Instruct | `us.meta.llama3-1-8b-instruct-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Meta Llama 3.1 70B Instruct | `us.meta.llama3-1-70b-instruct-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Meta Llama 3.1 405B Instruct | `us.meta.llama3-1-405b-instruct-v1:0` | us-east-2 | us-east-1, us-east-2, us-west-2 |
| US Meta Llama 3.2 1B Instruct | `us.meta.llama3-2-1b-instruct-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Meta Llama 3.2 3B Instruct | `us.meta.llama3-2-3b-instruct-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Meta Llama 3.2 11B Instruct | `us.meta.llama3-2-11b-instruct-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Meta Llama 3.2 90B Instruct | `us.meta.llama3-2-90b-instruct-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |
| US Meta Llama 3.3 70B Instruct | `us.meta.llama3-3-70b-instruct-v1:0` | us-east-1, us-east-2, us-west-2 | us-east-1, us-east-2, us-west-2 |

**Important Notes About Cross-Region Inference Profiles:**
- AWS will not add new regions to existing inference profiles; instead, new profiles may be created as regions are added
- Routing behavior can differ based on the source region (see Claude 3 Haiku example above)
- To check destination regions for a profile, you can use the `GetInferenceProfile` API with a control plane endpoint from your source region

##### Using Cross-Region Inference Profiles

To use a cross-region inference profile, simply specify the profile ID in your API call instead of a model ID:

```python
import boto3
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
inference_profile_id = 'us.anthropic.claude-3-sonnet-20240229-v1:0'

response = bedrock_runtime.invoke_model(
    modelId=inference_profile_id,
    body=json.dumps({
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000
    })
)
```

#### Application Inference Profiles

Application inference profiles are user-created profiles that can track metrics, costs, and usage for model invocation. You can create application inference profiles that route to a single region or to multiple regions using a cross-region inference profile.

##### Creating an Application Inference Profile from a Foundation Model

```python
import boto3
import json

# Initialize the Bedrock client
bedrock = boto3.client('bedrock', region_name='us-west-2')

# Create an application inference profile from a foundation model
response = bedrock.create_inference_profile(
    inferenceProfileName="MyClaudeSonnetProfile",
    modelSource={
        "copyFrom": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
    },
    description="Application inference profile for Claude 3 Sonnet",
    tags=[
        {
            "key": "project",
            "value": "customer-service-ai"
        },
        {
            "key": "department", 
            "value": "ai-research"
        }
    ]
)

print("Created inference profile with ARN:", response["inferenceProfileArn"])
print("Status:", response["status"])
```

##### Creating an Application Inference Profile from a Cross-Region Inference Profile

```python
import boto3
import json

# Initialize the Bedrock client
bedrock = boto3.client('bedrock', region_name='us-west-2')
account_id = "123456789012"  # Replace with your AWS account ID

# Create an application inference profile from a cross-region inference profile
response = bedrock.create_inference_profile(
    inferenceProfileName="MyCrossRegionClaudeSonnetProfile",
    modelSource={
        "copyFrom": f"arn:aws:bedrock:us-west-2:{account_id}:inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0"
    },
    description="Cross-region application inference profile for Claude 3 Sonnet",
    tags=[
        {
            "key": "environment",
            "value": "production"
        }
    ]
)

print("Created cross-region inference profile with ARN:", response["inferenceProfileArn"])
print("Status:", response["status"])
```

##### Using an Application Inference Profile with InvokeModel

```python
import boto3
import json

# Initialize the Bedrock Runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')
account_id = "123456789012"  # Replace with your AWS account ID

# Application inference profile ARN
inference_profile_arn = f"arn:aws:bedrock:us-west-2:{account_id}:inference-profile/MyClaudeSonnetProfile"

# Use the application inference profile for inference
response = bedrock_runtime.invoke_model(
    modelId=inference_profile_arn,
    body=json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain quantum computing in simple terms."
                    }
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.7
    })
)

# Process the response
response_body = json.loads(response['body'].read())
print(response_body['content'][0]['text'])
```

##### Using an Application Inference Profile with the Converse API

```python
import boto3
import json

# Initialize the Bedrock Runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')
account_id = "123456789012"  # Replace with your AWS account ID

# Application inference profile ARN
inference_profile_arn = f"arn:aws:bedrock:us-west-2:{account_id}:inference-profile/MyClaudeSonnetProfile"

# Use the application inference profile with the Converse API
response = bedrock_runtime.converse(
    modelId=inference_profile_arn,
    system=[{"text": "You are an expert in cloud computing."}],
    messages=[
        {
            "role": "user",
            "content": [{"text": "What are the main services offered by AWS?"}]
        }
    ]
)

# Process the response
print(response['output']['message']['content'][0]['text'])
```

##### Viewing Information About an Inference Profile

```python
import boto3

# Initialize the Bedrock client
bedrock = boto3.client('bedrock', region_name='us-west-2')
account_id = "123456789012"  # Replace with your AWS account ID

# Get information about the application inference profile
response = bedrock.get_inference_profile(
    inferenceProfileIdentifier=f"arn:aws:bedrock:us-west-2:{account_id}:inference-profile/MyClaudeSonnetProfile"
)

print("Inference Profile Name:", response.get("inferenceProfileName"))
print("Description:", response.get("description"))
print("Status:", response.get("status"))
print("Model Source:", response.get("modelSource"))
print("Creation Time:", response.get("creationTime"))
print("Last Modified Time:", response.get("lastModifiedTime"))
```

##### Listing Inference Profiles

```python
import boto3

# Initialize the Bedrock client
bedrock = boto3.client('bedrock', region_name='us-west-2')

# List all application inference profiles
response = bedrock.list_inference_profiles()

for profile in response.get("inferenceProfiles", []):
    print("Name:", profile.get("inferenceProfileName"))
    print("ARN:", profile.get("inferenceProfileArn"))
    print("Status:", profile.get("status"))
    print("Created:", profile.get("creationTime"))
    print("-" * 50)
```

##### Deleting an Application Inference Profile

```python
import boto3

# Initialize the Bedrock client
bedrock = boto3.client('bedrock', region_name='us-west-2')
account_id = "123456789012"  # Replace with your AWS account ID

# Delete the application inference profile
response = bedrock.delete_inference_profile(
    inferenceProfileIdentifier=f"arn:aws:bedrock:us-west-2:{account_id}:inference-profile/MyClaudeSonnetProfile"
)

print("Deletion status:", response.get("status"))
```

#### Using AWS CLI for Inference Profiles

You can also manage inference profiles using the AWS CLI:

##### Creating an Application Inference Profile with AWS CLI

```bash
# Create an application inference profile from a foundation model
aws bedrock create-inference-profile \
    --inference-profile-name MyClaudeSonnetProfile \
    --model-source '{"copyFrom": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"}' \
    --description "Application inference profile for Claude 3 Sonnet" \
    --tags '[{"key":"project","value":"customer-service-ai"}]' \
    --region us-west-2

# Create an application inference profile from a cross-region inference profile
aws bedrock create-inference-profile \
    --inference-profile-name MyCrossRegionClaudeSonnetProfile \
    --model-source '{"copyFrom": "arn:aws:bedrock:us-west-2:123456789012:inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0"}' \
    --description "Cross-region application inference profile for Claude 3 Sonnet" \
    --tags '[{"key":"environment","value":"production"}]' \
    --region us-west-2
```

#### Key Benefits of Using Inference Profiles

1. **Improved Throughput**: Cross-region inference profiles can route requests to multiple regions, increasing your overall throughput and reducing latency.

2. **Cost Tracking**: Application inference profiles allow you to attach tags for cost allocation, making it easier to track and analyze your spending on different AI projects.

3. **Usage Metrics**: Get detailed CloudWatch metrics for your model invocations when using application inference profiles.

4. **Resilience**: Cross-region inference profiles improve resilience by allowing requests to be routed to healthy regions if one region experiences issues.

5. **Simplified Resource Management**: Create a single resource (inference profile) that can be referenced across your applications instead of managing region-specific model endpoints.

6. **Higher Rate Limits**: Using cross-region inference profiles can help you achieve higher rate limits by leveraging quotas across multiple regions.

## Batch Inference

Batch inference allows processing multiple prompts asynchronously by sending requests in bulk and retrieving results from an Amazon S3 bucket.

### Supported Models and Regions

Not all models support batch inference. Below is a detailed breakdown of models that support batch inference and their regional availability.

#### Key Batch Inference Information

- **Pricing advantage**: Batch inference is offered at 50% of the On-Demand inference price for most models
- **Job limits**: There's a limit of 10 batch inference jobs per model per region
- **Provisioned models**: Batch inference is not supported for provisioned models
- **Core regions**: The primary regions supporting batch inference are us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, and eu-central-1

#### Models Supporting Application Inference Profiles

The following foundation models support application inference profiles, which can be used with both online and batch processing where available:

* Amazon Titan Embeddings G1 - Text
* Amazon Titan Image Generator G1 v2
* Amazon Titan Image Generator G1
* Amazon Titan Text Embeddings V2
* Anthropic Claude 2.1
* Anthropic Claude 3 Haiku
* Anthropic Claude 3 Opus
* Anthropic Claude 3 Sonnet
* Anthropic Claude 3.5 Sonnet
* Anthropic Claude 3.5 Haiku
* Anthropic Claude 3.7 Sonnet
* Meta Llama 3 70B Instruct
* Meta Llama 3 8B Instruct
* Meta Llama 3.2 11B Instruct
* Meta Llama 3.2 1B Instruct
* Meta Llama 3.2 3B Instruct
* Meta Llama 3.2 90B Instruct
* Mistral AI Mistral 7B Instruct
* Mistral AI Mixtral 8x7B Instruct
* Stability AI SDXL 1.0

#### Complete List of Models Supporting Batch Inference

The following table shows the complete and most up-to-date list of models that support batch inference and their specific regional availability:

| Provider   | Model                          | Regions Supporting Batch Inference                                                                                                                          |
| ---------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Amazon     | Nova Lite                      | us-east-1                                                                                                                                                    |
| Amazon     | Nova Micro                     | us-east-1                                                                                                                                                    |
| Amazon     | Nova Pro                       | us-east-1                                                                                                                                                    |
| Amazon     | Titan Text G1 - Express        | us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, eu-central-1                                                                                           |
| Amazon     | Titan Text G1 - Lite           | us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, eu-central-1                                                                                           |
| Amazon     | Titan Text G1 - Premier        | us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, eu-central-1                                                                                           |
| Amazon     | Titan Multimodal Embeddings G1 | us-east-1, us-west-2, ap-northeast-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1                     |
| Amazon     | Titan Text Embeddings V2       | us-east-1, us-west-2, ap-northeast-2, ca-central-1, eu-central-1, eu-west-2, sa-east-1                                                                       |
| Anthropic  | Claude 3.5 Haiku               | us-west-2                                                                                                                                                    |
| Anthropic  | Claude 3.5 Sonnet              | us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-southeast-1, eu-central-1                                                                           |
| Anthropic  | Claude 3.5 Sonnet v2           | us-west-2                                                                                                                                                    |
| Anthropic  | Claude 3.7 Sonnet              | us-east-1, us-east-2, us-west-2                                                                                                                              |
| Anthropic  | Claude 3 Haiku                 | us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1 |
| Anthropic  | Claude 3 Opus                  | us-west-2                                                                                                                                                    |
| Anthropic  | Claude 3 Sonnet                | us-east-1, us-west-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1                                      |
| Anthropic  | Claude 2.1                     | us-east-1, us-west-2, ap-northeast-1, eu-central-1                                                                                                           |
| Anthropic  | Claude 2.0                     | us-east-1, us-west-2, ap-southeast-1, eu-central-1                                                                                                           |
| Meta       | Llama 3.1 8B Instruct          | us-east-1, us-west-2                                                                                                                                         |
| Meta       | Llama 3.1 70B Instruct         | us-west-2                                                                                                                                                    |
| Meta       | Llama 3.1 405B Instruct        | us-west-2                                                                                                                                                    |
| Meta       | Llama 3.2 1B Instruct          | us-east-1, us-west-2                                                                                                                                         |
| Meta       | Llama 3.2 3B Instruct          | us-east-1, us-west-2                                                                                                                                         |
| Meta       | Llama 3.2 11B Instruct         | us-east-1, us-west-2                                                                                                                                         |
| Meta       | Llama 3.2 90B Instruct         | us-east-1, us-west-2                                                                                                                                         |
| Meta       | Llama 3.3 70B Instruct         | us-east-1, us-west-2                                                                                                                                         |
| AI21 Labs  | Jurassic-2 Mid                 | us-east-1                                                                                                                                                    |
| AI21 Labs  | Jurassic-2 Ultra               | us-east-1                                                                                                                                                    |
| AI21 Labs  | J2 Jumbo Instruct              | us-east-1                                                                                                                                                    |
| AI21 Labs  | Jamba-Instruct                 | us-east-1                                                                                                                                                    |
| AI21 Labs  | Jamba 1.5 Large                | us-east-1                                                                                                                                                    |
| AI21 Labs  | Jamba 1.5 Mini                 | us-east-1                                                                                                                                                    |
| Mistral AI | Mistral Small (24.02)          | us-east-1                                                                                                                                                    |
| Mistral AI | Mistral Large (24.02)          | us-east-1, us-west-2                                                                                                                                         |
| Mistral AI | Mistral Large (24.07)          | us-west-2                                                                                                                                                    |
| Mistral AI | Mixtral 8x7B Instruct          | us-east-1, us-west-2, ap-south-1, ap-southeast-2, ca-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1                                                   |
| Cohere     | Command                        | us-east-1, us-west-2                                                                                                                                         |
| Cohere     | Command R                      | us-east-1, us-west-2                                                                                                                                         |
| Cohere     | Command R+                     | us-east-1, us-west-2                                                                                                                                         |
| Cohere     | Embed English                  | us-east-1, us-west-2                                                                                                                                         |
| Cohere     | Embed Multilingual             | us-east-1, us-west-2                                                                                                                                         |

#### Detailed Batch Inference Model Availability

### Input/Output Formats

#### Input Format

Batch inference requires JSONL files in an S3 bucket. Each line contains a JSON object with:

1. A `recordId` field (optional, will be added by AWS if omitted)
2. A `modelInput` field containing the request body that matches the model's InvokeModel request format

Example JSONL file for Amazon Titan Text model:
```json
{ "recordId": "12345abcdef", "modelInput": { "inputText": "Write a poem about sunshine", "textGenerationConfig": { "temperature": 0.7, "maxTokenCount": 512 } } }
{ "recordId": "67890ghijkl", "modelInput": { "inputText": "Write a short story about the ocean", "textGenerationConfig": { "temperature": 0.8, "maxTokenCount": 1024 } } }
```

Example JSONL file for Anthropic Claude:
```json
{ "recordId": "record1", "modelInput": { "messages": [{ "role": "user", "content": [{ "text": "Explain quantum computing" }] }], "anthropic_version": "bedrock-2023-05-31", "max_tokens": 1000 } }
{ "recordId": "record2", "modelInput": { "messages": [{ "role": "user", "content": [{ "text": "Write a function to calculate prime numbers" }] }], "anthropic_version": "bedrock-2023-05-31", "max_tokens": 1000 } }
```

#### Output Format

The output is stored in the specified S3 location and contains:
- A response file for each input record
- Each file contains the model's response in the same format as the corresponding InvokeModel API response

### Batch Job Creation and Management

#### Creating a Batch Inference Job

**Using the AWS SDK:**
```python
import boto3

bedrock_client = boto3.client('bedrock')

response = bedrock_client.create_model_invocation_job(
    jobName="my-batch-job",
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    roleArn="arn:aws:iam::123456789012:role/BedrockBatchJobRole",
    inputDataConfig={
        "s3InputDataConfig": {
            "s3Uri": "s3://my-input-bucket/batch-inputs/"
        }
    },
    outputDataConfig={
        "s3OutputDataConfig": {
            "s3Uri": "s3://my-output-bucket/batch-outputs/"
        }
    }
)

job_arn = response['jobArn']
```

#### Comprehensive Batch Inference Job Management

Here are detailed examples for managing batch inference jobs throughout their lifecycle:

##### Creating a Batch Job with Advanced Options

```python
import boto3
import json

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock', region_name='us-west-2')

# Create a batch inference job with advanced options
response = bedrock_client.create_model_invocation_job(
    # Required parameters
    jobName="example-batch-job",
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    roleArn="arn:aws:iam::123456789012:role/BedrockBatchJobRole",
    inputDataConfig={
        "s3InputDataConfig": {
            "s3Uri": "s3://my-input-bucket/batch-data/",
            # Optional - specify KMS key for input encryption
            "kmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab"
        }
    },
    outputDataConfig={
        "s3OutputDataConfig": {
            "s3Uri": "s3://my-output-bucket/batch-results/",
            # Optional - specify KMS key for output encryption
            "kmsKeyId": "arn:aws:kms:us-west-2:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab"
        }
    },
    
    # Optional parameters
    # Timeout in hours (default is 24 hours)
    timeoutDurationInHours=12,
    
    # VPC configuration for enhanced security
    vpcConfig={
        "subnetIds": [
            "subnet-0abc123def456789a",
            "subnet-0def456789abc123d"
        ],
        "securityGroupIds": [
            "sg-0123456789abcdef0"
        ]
    },
    
    # Tags for resource organization and cost tracking
    tags=[
        {
            "key": "Department",
            "value": "AI-Research"
        },
        {
            "key": "Project",
            "value": "TextSummarization"
        },
        {
            "key": "Environment",
            "value": "Production"
        }
    ],
    
    # Ensuring idempotency
    clientRequestToken="unique-token-123456789"
)

print(f"Batch job created successfully. Job ARN: {response['jobArn']}")
print(f"Job status: {response['status']}")
```

##### Get Batch Inference Job Status

```python
import boto3

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock', region_name='us-west-2')

# Get details about a specific batch inference job
response = bedrock_client.get_model_invocation_job(
    jobIdentifier="arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/example-batch-job"
)

# Print job details
print(f"Job Name: {response['jobName']}")
print(f"Model ID: {response['modelId']}")
print(f"Status: {response['status']}")
```

##### List All Batch Inference Jobs

```python
import boto3

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock', region_name='us-west-2')

# List all batch inference jobs with optional filters
response = bedrock_client.list_model_invocation_jobs(
    # Optional filters
    # maxResults=10,
    # nextToken="string",
    # filters=[
    #     {
    #         "name": "ModelId",
    #         "operator": "Equals",
    #         "values": ["anthropic.claude-3-haiku-20240307-v1:0"]
    #     },
    #     {
    #         "name": "Status",
    #         "operator": "Equals",
    #         "values": ["Completed"]
    #     }
    # ]
)

# Print information about each job
for job in response['modelInvocationJobs']:
    print(f"Job Name: {job['jobName']}")
    print(f"Job ARN: {job['jobArn']}")
    print(f"Model ID: {job['modelId']}")
    print(f"Status: {job['status']}")
    print(f"Creation Time: {job['creationTime']}")
    if 'endTime' in job:
        print(f"End Time: {job['endTime']}")
    print("-" * 50)

# Handle pagination
if 'nextToken' in response:
    print("More results available. Use nextToken for pagination.")
```

##### Stop a Batch Inference Job

```python
import boto3

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock', region_name='us-west-2')

# Stop a running batch inference job
response = bedrock_client.stop_model_invocation_job(
    jobIdentifier="arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/example-batch-job"
)

print(f"Job stopping initiated. Current status: {response['status']}")
```

##### Using AWS CLI for Batch Inference Jobs

```bash
# Create a batch inference job
aws bedrock create-model-invocation-job \
    --job-name example-batch-job \
    --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
    --role-arn arn:aws:iam::123456789012:role/BedrockBatchJobRole \
    --input-data-config '{"s3InputDataConfig":{"s3Uri":"s3://my-input-bucket/batch-data/"}}' \
    --output-data-config '{"s3OutputDataConfig":{"s3Uri":"s3://my-output-bucket/batch-results/"}}' \
    --region us-west-2

# Get status of a batch inference job
aws bedrock get-model-invocation-job \
    --job-identifier arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/example-batch-job \
    --region us-west-2

# List all batch inference jobs
aws bedrock list-model-invocation-jobs \
    --region us-west-2

# Stop a batch inference job
aws bedrock stop-model-invocation-job \
    --job-identifier arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/example-batch-job \
    --region us-west-2
```

#### Batch Inference Job Status Monitoring

Batch inference jobs can have the following status values:

| Status | Description |
|--------|-------------|
| `Submitted` | Job has been submitted but processing hasn't started yet |
| `InProgress` | Job is currently being processed |
| `Completed` | Job has completed successfully |
| `Failed` | Job has failed due to an error |
| `Stopping` | Job is in the process of stopping |
| `Stopped` | Job has been stopped by the user |
| `Expired` | Job has reached its timeout duration |

#### Best Practices for Batch Inference Jobs

1. **Use appropriate timeouts**: Set `timeoutDurationInHours` based on your expected job duration.
2. **Monitor job status**: Regularly poll job status to check for completion or failures.
3. **Implement error handling**: Check for failure messages in job responses and implement appropriate retry logic.
4. **Structure your input data efficiently**: Group similar requests together for better processing efficiency.
5. **Consider VPC configuration**: For sensitive data, use VPC configurations to enhance security.
6. **Add tags for cost allocation**: Use tags to track costs across different projects or departments.
7. **Limit concurrent jobs**: Be aware of the quota limit of 10 batch inference jobs per model per region.
8. **Take advantage of pricing benefits**: Batch inference is offered at 50% of the On-Demand inference price.

### Input/Output Formats

// ... existing code ...

## Cross-Region Considerations

### Inference Profiles for Cross-Region Requests

AWS Bedrock supports cross-region inference to increase throughput and improve resilience:

1. **System-defined inference profiles**: Pre-configured by AWS to route requests across multiple regions
2. **Custom application inference profiles**: Can be configured to use system-defined profiles

### Data Sharing Warning

When using cross-region inference, data may be shared across regions. Consider data residency requirements before using cross-region inference.

### Accessing Models in Different Regions

Some models are only available in specific regions. Cross-region inference allows accessing these models from other supported regions.

## Authentication and Permissions

### Required IAM Permissions

For online processing:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "*"
        }
    ]
}
```

For batch inference:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelInvocationJob",
                "bedrock:GetModelInvocationJob",
                "bedrock:StopModelInvocationJob",
                "bedrock:ListModelInvocationJobs"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::input-bucket",
                "arn:aws:s3:::input-bucket/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::output-bucket",
                "arn:aws:s3:::output-bucket/*"
            ]
        }
    ]
}
```

## Error Handling

Common errors to handle:

1. **AccessDeniedException**: Insufficient permissions
2. **ModelErrorException**: Error during model processing
3. **ModelNotReadyException**: Model not ready to serve inference
4. **ModelTimeoutException**: Request took too long to process
5. **ResourceNotFoundException**: Specified resource not found
6. **ServiceQuotaExceededException**: Request exceeds service quota
7. **ThrottlingException**: Request denied due to rate limiting
8. **ValidationException**: Input fails to satisfy constraints

## Monitoring and Logging

### CloudWatch Metrics

AWS Bedrock publishes metrics to CloudWatch, including:
- Invocation count
- Invocation latency
- Token usage (input and output)
- Error rates

### Cost Tracking

Use tags on inference profiles to track costs by project, department, or application.

### Best Practices

1. Set up CloudWatch alarms for quota limits
2. Monitor token usage for cost control
3. Use tags for cost allocation
4. Implement detailed logging for debugging 