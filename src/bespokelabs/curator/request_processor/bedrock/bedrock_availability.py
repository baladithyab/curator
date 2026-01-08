"""Bedrock model and regional availability information.

This module provides awareness of which models are available in which regions,
and which APIs (Converse vs InvokeModel) are supported for each model.

Supports both dynamic discovery via boto3 APIs and static fallbacks.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from bespokelabs.curator.log import logger

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

# Cross-region inference profile prefixes
CROSS_REGION_PREFIXES: Dict[str, List[str]] = {
    "us.": ["us-east-1", "us-east-2", "us-west-1", "us-west-2"],
    "eu.": ["eu-central-1", "eu-west-1", "eu-west-2", "eu-west-3", "eu-north-1"],
    "apac.": ["ap-northeast-1", "ap-northeast-2", "ap-south-1", "ap-southeast-1", "ap-southeast-2"],
}


@dataclass
class BedrockModelInfo:
    """Information about a Bedrock model."""

    model_id: str
    model_name: str
    provider: str
    input_modalities: List[str] = field(default_factory=list)
    output_modalities: List[str] = field(default_factory=list)
    inference_types: List[str] = field(default_factory=list)
    streaming_supported: bool = False
    customizations_supported: List[str] = field(default_factory=list)
    response_streaming_supported: bool = False

    @property
    def supports_text_input(self) -> bool:
        """Check if model supports text input."""
        return "TEXT" in self.input_modalities

    @property
    def supports_image_input(self) -> bool:
        """Check if model supports image input."""
        return "IMAGE" in self.input_modalities

    @property
    def supports_text_output(self) -> bool:
        """Check if model supports text output."""
        return "TEXT" in self.output_modalities

    @property
    def is_embedding_model(self) -> bool:
        """Check if this is an embedding model."""
        return "EMBEDDING" in self.output_modalities

    @property
    def is_image_generation_model(self) -> bool:
        """Check if this is an image generation model."""
        return "IMAGE" in self.output_modalities and "TEXT" not in self.output_modalities


class BedrockAvailabilityManager:
    """Manages Bedrock model availability information with dynamic discovery."""

    def __init__(self, region: Optional[str] = None, profile: Optional[str] = None):
        """Initialize the availability manager.

        Args:
            region: AWS region to query. Defaults to AWS_REGION or us-east-1.
            profile: AWS profile to use. Defaults to AWS_PROFILE.
        """
        self.region = region or os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self.profile = profile or os.getenv("AWS_PROFILE")
        self._bedrock_client = None
        self._models_cache: Dict[str, BedrockModelInfo] = {}
        self._converse_supported_cache: Dict[str, bool] = {}

    @property
    def bedrock_client(self):
        """Lazily initialize the Bedrock control plane client."""
        if self._bedrock_client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "boto3 is required for Bedrock support. Install with: pip install boto3"
                ) from e

            session_kwargs = {}
            if self.profile:
                session_kwargs["profile_name"] = self.profile

            session = boto3.Session(**session_kwargs)
            self._bedrock_client = session.client(
                "bedrock",
                region_name=self.region,
            )
        return self._bedrock_client

    def list_foundation_models(self, refresh: bool = False) -> Dict[str, BedrockModelInfo]:
        """List all available foundation models in the current region.

        Args:
            refresh: Force refresh the cache

        Returns:
            Dictionary mapping model_id to BedrockModelInfo
        """
        if self._models_cache and not refresh:
            return self._models_cache

        try:
            response = self.bedrock_client.list_foundation_models()
            models = response.get("modelSummaries", [])

            for model in models:
                model_id = model.get("modelId", "")
                info = BedrockModelInfo(
                    model_id=model_id,
                    model_name=model.get("modelName", ""),
                    provider=model.get("providerName", ""),
                    input_modalities=model.get("inputModalities", []),
                    output_modalities=model.get("outputModalities", []),
                    inference_types=model.get("inferenceTypesSupported", []),
                    streaming_supported=model.get("responseStreamingSupported", False),
                    customizations_supported=model.get("customizationsSupported", []),
                )
                self._models_cache[model_id] = info

            logger.debug(f"Loaded {len(self._models_cache)} Bedrock models from API")

        except Exception as e:
            logger.warning(f"Failed to list Bedrock models: {e}. Using static fallbacks.")

        return self._models_cache

    def get_model_info(self, model_id: str) -> Optional[BedrockModelInfo]:
        """Get detailed information about a specific model.

        Args:
            model_id: The Bedrock model ID

        Returns:
            BedrockModelInfo or None if not found
        """
        # Check cache first
        if model_id in self._models_cache:
            return self._models_cache[model_id]

        # Try to fetch from API
        try:
            response = self.bedrock_client.get_foundation_model(modelIdentifier=model_id)
            details = response.get("modelDetails", {})

            info = BedrockModelInfo(
                model_id=details.get("modelId", model_id),
                model_name=details.get("modelName", ""),
                provider=details.get("providerName", ""),
                input_modalities=details.get("inputModalities", []),
                output_modalities=details.get("outputModalities", []),
                inference_types=details.get("inferenceTypesSupported", []),
                streaming_supported=details.get("responseStreamingSupported", False),
                customizations_supported=details.get("customizationsSupported", []),
            )
            self._models_cache[model_id] = info
            return info

        except Exception as e:
            logger.debug(f"Could not get model info for {model_id}: {e}")
            return None

    def supports_converse_api(self, model_id: str) -> bool:
        """Check if a model supports the Converse API.

        Uses dynamic discovery when possible, with static fallbacks.

        Args:
            model_id: The Bedrock model ID

        Returns:
            True if model supports Converse API
        """
        # Check cache
        if model_id in self._converse_supported_cache:
            return self._converse_supported_cache[model_id]

        # Get model info
        model_info = self.get_model_info(model_id)

        if model_info:
            # Converse API doesn't support embedding or image generation models
            if model_info.is_embedding_model or model_info.is_image_generation_model:
                self._converse_supported_cache[model_id] = False
                return False

            # Text-to-text models generally support Converse
            if model_info.supports_text_input and model_info.supports_text_output:
                self._converse_supported_cache[model_id] = True
                return True

        # Fall back to static checks
        result = _static_supports_converse_api(model_id)
        self._converse_supported_cache[model_id] = result
        return result

    def requires_invoke_model(self, model_id: str) -> bool:
        """Check if a model requires InvokeModel API.

        Args:
            model_id: The Bedrock model ID

        Returns:
            True if model must use InvokeModel
        """
        model_info = self.get_model_info(model_id)

        if model_info:
            # Embedding and image generation models require InvokeModel
            return model_info.is_embedding_model or model_info.is_image_generation_model

        # Fall back to static check
        return _static_requires_invoke_model(model_id)

    def is_batch_available_in_region(self, model_id: str, region: Optional[str] = None) -> bool:
        """Check if batch inference is available for a model in a region.

        Args:
            model_id: The Bedrock model ID
            region: AWS region (defaults to current region)

        Returns:
            True if batch inference is available
        """
        target_region = region or self.region

        # Check if region supports batch at all
        if target_region not in BATCH_INFERENCE_REGIONS:
            return False

        # Try to get model info to check inference types
        model_info = self.get_model_info(model_id)
        if model_info and "ON_DEMAND" in model_info.inference_types:
            # ON_DEMAND typically supports batch
            return True

        # Fall back to static check
        return _static_is_batch_available(model_id, target_region)

    def get_available_models_for_batch(self, region: Optional[str] = None) -> List[str]:
        """Get list of models available for batch inference in a region.

        Args:
            region: AWS region (defaults to current region)

        Returns:
            List of model IDs that support batch inference
        """
        target_region = region or self.region

        if target_region not in BATCH_INFERENCE_REGIONS:
            return []

        # Get all models and filter
        models = self.list_foundation_models()
        batch_models = []

        for model_id, info in models.items():
            if "ON_DEMAND" in info.inference_types:
                batch_models.append(model_id)

        return batch_models


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


# ============================================================================
# Static fallbacks when API discovery is not available
# ============================================================================

# Models that support the Converse API (static fallback)
_STATIC_CONVERSE_SUPPORTED: Set[str] = {
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
    # Cohere (Command R/R+ only)
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

# Models that require InvokeModel (static fallback)
_STATIC_INVOKE_MODEL_ONLY: Set[str] = {
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

# Static batch model region mapping
_STATIC_BATCH_MODEL_REGIONS: Dict[str, List[str]] = {
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
    "amazon.nova-lite-v1:0": ["me-central-1", "us-east-1", "us-gov-west-1"],
    "amazon.nova-micro-v1:0": ["us-east-1", "us-gov-west-1"],
    "amazon.nova-pro-v1:0": ["ap-southeast-3", "me-central-1", "us-east-1", "us-gov-west-1"],
}


def _static_supports_converse_api(model_id: str) -> bool:
    """Static check for Converse API support."""
    if model_id in _STATIC_CONVERSE_SUPPORTED:
        return True

    # Check prefix match for versioned models
    base_model = model_id.split(":")[0] if ":" in model_id else model_id
    for supported in _STATIC_CONVERSE_SUPPORTED:
        supported_base = supported.split(":")[0] if ":" in supported else supported
        if base_model == supported_base:
            return True

    # Check provider patterns - most chat models support Converse
    if any(prefix in model_id for prefix in ["anthropic.claude", "amazon.nova", "meta.llama", "mistral."]):
        if not any(x in model_id.lower() for x in ["embed", "image", "canvas", "reel"]):
            return True

    return False


def _static_requires_invoke_model(model_id: str) -> bool:
    """Static check for models requiring InvokeModel."""
    if model_id in _STATIC_INVOKE_MODEL_ONLY:
        return True

    # Pattern-based detection
    if any(x in model_id.lower() for x in ["embed", "image", "canvas", "reel", "diffusion"]):
        return True

    return False


def _static_is_batch_available(model_id: str, region: str) -> bool:
    """Static check for batch availability."""
    if region not in BATCH_INFERENCE_REGIONS:
        return False

    if model_id in _STATIC_BATCH_MODEL_REGIONS:
        return region in _STATIC_BATCH_MODEL_REGIONS[model_id]

    # Check for cross-region profiles
    for prefix, regions in CROSS_REGION_PREFIXES.items():
        if model_id.startswith(prefix) and region in regions:
            return True

    return False


# ============================================================================
# Module-level convenience functions (use cached singleton)
# ============================================================================

_availability_manager: Optional[BedrockAvailabilityManager] = None


def get_availability_manager(
    region: Optional[str] = None,
    profile: Optional[str] = None,
    refresh: bool = False,
) -> BedrockAvailabilityManager:
    """Get the singleton availability manager instance.

    Args:
        region: AWS region
        profile: AWS profile
        refresh: Force create a new instance

    Returns:
        BedrockAvailabilityManager instance
    """
    global _availability_manager

    if _availability_manager is None or refresh:
        _availability_manager = BedrockAvailabilityManager(region=region, profile=profile)

    return _availability_manager


def supports_converse_api(model_id: str) -> bool:
    """Check if a model supports the Converse API.

    Args:
        model_id: The Bedrock model ID

    Returns:
        True if model supports Converse API
    """
    try:
        manager = get_availability_manager()
        return manager.supports_converse_api(model_id)
    except Exception:
        # Fall back to static if boto3 not available
        return _static_supports_converse_api(model_id)


def has_limited_converse_support(model_id: str) -> bool:
    """Check if a model has limited Converse support (no chat history).

    Args:
        model_id: The Bedrock model ID

    Returns:
        True if model has limited Converse support
    """
    # These models can only handle one user message at a time
    limited_models = {
        "ai21.j2-ultra-v1",
        "ai21.j2-mid-v1",
        "cohere.command-text-v14",
        "cohere.command-light-text-v14",
    }
    return model_id in limited_models


def requires_invoke_model(model_id: str) -> bool:
    """Check if a model requires InvokeModel API.

    Args:
        model_id: The Bedrock model ID

    Returns:
        True if model must use InvokeModel
    """
    try:
        manager = get_availability_manager()
        return manager.requires_invoke_model(model_id)
    except Exception:
        return _static_requires_invoke_model(model_id)


def is_batch_available(model_id: str, region: str) -> bool:
    """Check if batch inference is available for a model in a region.

    Args:
        model_id: The Bedrock model ID
        region: AWS region code

    Returns:
        True if batch inference is available
    """
    try:
        manager = get_availability_manager()
        return manager.is_batch_available_in_region(model_id, region)
    except Exception:
        return _static_is_batch_available(model_id, region)


def get_available_batch_regions(model_id: str) -> List[str]:
    """Get list of regions where batch inference is available for a model.

    Args:
        model_id: The Bedrock model ID

    Returns:
        List of region codes where batch is available
    """
    if model_id in _STATIC_BATCH_MODEL_REGIONS:
        return _STATIC_BATCH_MODEL_REGIONS[model_id]

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
