"""Bedrock model and regional availability information.

This module provides awareness of which models are available in which regions,
and which APIs (Converse vs InvokeModel) are supported for each model.
"""

from typing import Dict, List, Optional, Set

# Models that support the Converse API (preferred unified interface)
# Based on https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html
CONVERSE_SUPPORTED_MODELS: Set[str] = {
    # AI21 Labs
    "ai21.jamba-instruct-v1:0",
    "ai21.jamba-1-5-large-v1:0",
    "ai21.jamba-1-5-mini-v1:0",
    # Amazon Nova
    "amazon.nova-premier-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-2-lite-v1:0",
    # Amazon Titan
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1",
    "amazon.titan-text-premier-v1:0",
    # Anthropic Claude
    "anthropic.claude-v2",
    "anthropic.claude-v2:1",
    "anthropic.claude-instant-v1",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-opus-4-20250805-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-5-20251101-v1:0",
    # Cohere (Command R/R+ only, not Command/Command Light for chat)
    "cohere.command-r-v1:0",
    "cohere.command-r-plus-v1:0",
    # DeepSeek
    "deepseek.r1-v1:0",
    "deepseek.v3-v1:0",
    # Meta Llama
    "meta.llama2-13b-chat-v1",
    "meta.llama2-70b-chat-v1",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-2-1b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-2-11b-instruct-v1:0",
    "meta.llama3-2-90b-instruct-v1:0",
    "meta.llama3-3-70b-instruct-v1:0",
    "meta.llama4-maverick-17b-instruct-v1:0",
    "meta.llama4-scout-17b-instruct-v1:0",
    # Mistral AI
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-large-2407-v1:0",
    "mistral.mistral-small-2402-v1:0",
    "mistral.pixtral-large-2502-v1:0",
    # Writer
    "writer.palmyra-x4-v1:0",
    "writer.palmyra-x5-v1:0",
}

# Models that have LIMITED Converse support (no chat/conversation history)
CONVERSE_LIMITED_MODELS: Set[str] = {
    "ai21.j2-ultra-v1",
    "ai21.j2-mid-v1",
    "cohere.command-text-v14",
    "cohere.command-light-text-v14",
}

# Models that do NOT support Converse API (must use InvokeModel)
INVOKE_MODEL_ONLY: Set[str] = {
    # Embedding models
    "amazon.titan-embed-text-v1",
    "amazon.titan-embed-text-v2:0",
    "amazon.titan-embed-image-v1",
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
    # Image generation models
    "amazon.titan-image-generator-v1",
    "amazon.titan-image-generator-v2:0",
    "amazon.nova-canvas-v1:0",
    "stability.stable-diffusion-xl-v1",
    # Video generation
    "amazon.nova-reel-v1:0",
    # Custom imported GPT-OSS models
    "openai.gpt-oss-120b-1:0",
    "openai.gpt-oss-20b-1:0",
}

# Regions where Bedrock is available
BEDROCK_REGIONS: List[str] = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-southeast-5",
    "ap-southeast-7",
    "ca-central-1",
    "ca-west-1",
    "eu-central-1",
    "eu-central-2",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "il-central-1",
    "me-central-1",
    "me-south-1",
    "sa-east-1",
    "us-gov-east-1",
    "us-gov-west-1",
]

# Regions where batch inference is available
BATCH_INFERENCE_REGIONS: List[str] = [
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "ap-northeast-1",
    "ap-south-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-south-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "me-central-1",
    "sa-east-1",
    "us-gov-west-1",
]

# Model to region availability mapping for batch inference
# Key models with their single-region batch support
BATCH_MODEL_REGIONS: Dict[str, List[str]] = {
    # Anthropic Claude
    "anthropic.claude-3-haiku-20240307-v1:0": [
        "ap-northeast-1", "ap-northeast-2", "ap-south-1", "ap-southeast-1",
        "ap-southeast-2", "ca-central-1", "eu-central-1", "eu-central-2",
        "eu-west-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-east-1", "us-west-2"
    ],
    "anthropic.claude-3-opus-20240229-v1:0": ["us-west-2"],
    "anthropic.claude-3-sonnet-20240229-v1:0": [
        "ap-northeast-2", "ap-south-1", "ap-southeast-2", "ca-central-1",
        "eu-central-1", "eu-west-1", "eu-west-2", "eu-west-3", "sa-east-1",
        "us-east-1", "us-west-2"
    ],
    "anthropic.claude-3-5-haiku-20241022-v1:0": ["us-west-2"],
    "anthropic.claude-3-5-sonnet-20240620-v1:0": [
        "ap-northeast-1", "ap-northeast-2", "ap-southeast-1", "eu-central-1",
        "us-east-1", "us-east-2", "us-west-2"
    ],
    "anthropic.claude-3-5-sonnet-20241022-v2:0": ["us-west-2"],
    # Amazon Nova
    "amazon.nova-lite-v1:0": ["me-central-1", "us-east-1", "us-gov-west-1"],
    "amazon.nova-micro-v1:0": ["us-east-1", "us-gov-west-1"],
    "amazon.nova-pro-v1:0": ["ap-southeast-3", "me-central-1", "us-east-1", "us-gov-west-1"],
    # Meta Llama
    "meta.llama3-1-405b-instruct-v1:0": ["us-west-2"],
    "meta.llama3-1-70b-instruct-v1:0": ["us-west-2"],
    "meta.llama3-1-8b-instruct-v1:0": ["us-west-2"],
    "meta.llama3-3-70b-instruct-v1:0": ["us-east-2"],
    # Mistral
    "mistral.mistral-large-2407-v1:0": ["us-west-2"],
    "mistral.mistral-small-2402-v1:0": ["us-east-1"],
}

# Cross-region inference profile prefixes
CROSS_REGION_PREFIXES: Dict[str, List[str]] = {
    "us.": ["us-east-1", "us-east-2", "us-west-1", "us-west-2"],
    "eu.": ["eu-central-1", "eu-west-1", "eu-west-2", "eu-west-3", "eu-north-1"],
    "apac.": ["ap-northeast-1", "ap-northeast-2", "ap-south-1", "ap-southeast-1", "ap-southeast-2"],
}


def supports_converse_api(model_id: str) -> bool:
    """Check if a model supports the Converse API.

    Args:
        model_id: The Bedrock model ID

    Returns:
        True if model supports Converse API, False otherwise
    """
    # Check exact match
    if model_id in CONVERSE_SUPPORTED_MODELS:
        return True

    # Check prefix match for versioned models
    base_model = model_id.split(":")[0] if ":" in model_id else model_id
    for supported in CONVERSE_SUPPORTED_MODELS:
        supported_base = supported.split(":")[0] if ":" in supported else supported
        if base_model == supported_base:
            return True

    return False


def has_limited_converse_support(model_id: str) -> bool:
    """Check if a model has limited Converse support (no chat history).

    Args:
        model_id: The Bedrock model ID

    Returns:
        True if model has limited Converse support
    """
    return model_id in CONVERSE_LIMITED_MODELS


def requires_invoke_model(model_id: str) -> bool:
    """Check if a model requires InvokeModel API (doesn't support Converse).

    Args:
        model_id: The Bedrock model ID

    Returns:
        True if model must use InvokeModel
    """
    if model_id in INVOKE_MODEL_ONLY:
        return True

    # Embedding and image models don't support Converse
    if any(x in model_id.lower() for x in ["embed", "image", "canvas", "reel", "diffusion"]):
        return True

    return False


def is_batch_available(model_id: str, region: str) -> bool:
    """Check if batch inference is available for a model in a region.

    Args:
        model_id: The Bedrock model ID
        region: AWS region code

    Returns:
        True if batch inference is available
    """
    if region not in BATCH_INFERENCE_REGIONS:
        return False

    if model_id in BATCH_MODEL_REGIONS:
        return region in BATCH_MODEL_REGIONS[model_id]

    # For cross-region inference profiles, check the prefix
    for prefix, regions in CROSS_REGION_PREFIXES.items():
        if model_id.startswith(prefix) and region in regions:
            return True

    return False


def get_available_batch_regions(model_id: str) -> List[str]:
    """Get list of regions where batch inference is available for a model.

    Args:
        model_id: The Bedrock model ID

    Returns:
        List of region codes where batch is available
    """
    if model_id in BATCH_MODEL_REGIONS:
        return BATCH_MODEL_REGIONS[model_id]

    # Check for cross-region profiles
    for prefix, regions in CROSS_REGION_PREFIXES.items():
        if model_id.startswith(prefix):
            return [r for r in regions if r in BATCH_INFERENCE_REGIONS]

    return []


def get_model_provider(model_id: str) -> Optional[str]:
    """Extract the provider name from a model ID.

    Args:
        model_id: The Bedrock model ID

    Returns:
        Provider name (e.g., 'anthropic', 'amazon', 'meta')
    """
    if "." in model_id:
        return model_id.split(".")[0]
    return None


def is_cross_region_profile(model_id: str) -> bool:
    """Check if a model ID is a cross-region inference profile.

    Args:
        model_id: The Bedrock model ID

    Returns:
        True if it's a cross-region inference profile
    """
    return any(model_id.startswith(prefix) for prefix in CROSS_REGION_PREFIXES.keys())
