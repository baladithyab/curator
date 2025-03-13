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

#### Creating and Using Inference Profiles

**Creating a single-region application inference profile:**
```python
import boto3
bedrock_client = boto3.client("bedrock", region_name="us-east-1")
response = bedrock_client.create_inference_profile(
    inferenceProfileName="my-inference-profile",
    modelSource={
        "copyFrom": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
    },
    description="My inference profile for Claude 3 Sonnet",
    tags=[
        {
            "key": "project",
            "value": "my-project"
        }
    ]
)
```

**Using an inference profile with the Converse API:**
```python
import boto3
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
inference_profile_id = 'us.anthropic.claude-3-sonnet-20240229-v1:0'

response = bedrock_runtime.converse(
    modelId=inference_profile_id,
    system=[{"text": "You are an expert on AWS AI services."}],
    messages=[{
        "role": "user",
        "content": [{"text": "Tell me about AWS Bedrock"}]
    }]
)
```

### API Reference

AWS Bedrock provides multiple APIs for online processing:

1. **InvokeModel**: Synchronous single prompt-response (non-streaming)
2. **InvokeModelWithResponseStream**: Streaming response for single prompt
3. **Converse**: Conversation-based API with messages
4. **ConverseStream**: Streaming version of the Converse API

### Implementation Considerations

1. **Rate Limiting**: AWS Bedrock enforces rate limits based on tokens per minute and requests per minute
2. **Timeouts**: Model responses may timeout for complex queries
3. **Error Handling**: Implement retries for throttling errors
4. **Token Usage**: Monitor token usage for cost management
5. **Model Selection**: Choose appropriate model based on capabilities and region availability

## Batch Inference

Batch inference allows processing multiple prompts asynchronously by sending requests in bulk and retrieving results from an Amazon S3 bucket.

### Supported Models and Regions

Not all models support batch inference. Below is a detailed breakdown of models that support batch inference and their regional availability.

#### Key Batch Inference Information

- **Pricing advantage**: Batch inference is offered at 50% of the On-Demand inference price for most models
- **Job limits**: There's a limit of 10 batch inference jobs per model per region
- **Provisioned models**: Batch inference is not supported for provisioned models
- **Core regions**: The primary regions supporting batch inference are us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, and eu-central-1

#### Detailed Batch Inference Model Availability

**Amazon Models**:
- Amazon Titan Text models (Express, Lite, Premier): us-east-1, us-west-2, ap-northeast-1, ap-southeast-1, eu-central-1
- Amazon Titan Embeddings models: Available in selected regions similar to their online processing availability
- Amazon Nova models: us-east-1 only

**Anthropic Claude Models**:
- Claude 3 Opus: us-west-2 only
- Claude 3 Sonnet: us-east-1, us-west-2, ap-south-1, ap-southeast-2, ca-central-1, eu-central-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1
- Claude 3 Haiku: Available in all regions where online processing is supported
- Claude 3.5 Sonnet: us-east-1, us-west-2, ap-northeast-1, ap-northeast-2, ap-southeast-1, eu-central-1
- Claude 3.5 Sonnet v2: us-west-2 only
- Claude 3.5 Haiku: us-west-2 only
- Claude 3.7 Sonnet: us-east-1, us-east-2, us-west-2

**AI21 Labs Models**:
- Jurassic-2 Models: us-east-1 only
- Jamba Models: us-east-1, us-west-2

**Meta Llama Models**:
- Llama 3.1 405B Instruct: us-west-2 only
- Llama 3.1 70B Instruct: us-west-2
- Other Llama models: Primarily available in us-east-1, us-west-2

**Mistral AI and Cohere Models**:
- Primarily available in us-east-1, us-west-2, with some availability in other regions

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

#### Monitoring Batch Jobs

```python
response = bedrock_client.get_model_invocation_job(
    jobIdentifier=job_arn
)

status = response['status']
print(f"Job status: {status}")

# Status will be one of: IN_PROGRESS, COMPLETED, FAILED, STOPPING, STOPPED
```

#### Stopping a Batch Job

```python
response = bedrock_client.stop_model_invocation_job(
    jobIdentifier=job_arn
)
```

### Implementation Considerations

1. **Quota Limits**: Batch inference has specific quotas to consider:
   - Minimum records per job
   - Maximum records per input file
   - Maximum records per job
   - Maximum input file size
   - Maximum cumulative job size

2. **IAM Permissions**: The IAM role requires permissions for:
   - S3 read access to input bucket
   - S3 write access to output bucket
   - Bedrock model invocation permissions

3. **VPC Configuration**: Batch jobs can be configured to run within a VPC for enhanced security

4. **Timeouts**: Batch jobs have a configurable timeout (in hours)

5. **Cost Management**: Monitor batch job costs, as they accumulate based on the number of tokens processed

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