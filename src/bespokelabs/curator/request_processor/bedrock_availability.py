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
class BedrockQuotas:
    """Bedrock service quotas for batch inference."""

    max_records_per_batch_job: int = 50_000
    min_records_per_batch_job: int = 1
    max_records_per_input_file: int = 50_000
    max_input_file_size_bytes: int = 1 * 1024 * 1024 * 1024  # 1 GB
    max_batch_job_size_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB
    max_concurrent_batch_jobs: int = 10

    @classmethod
    def get_default(cls) -> "BedrockQuotas":
        """Get default quota values."""
        return cls()


@dataclass
class BedrockModelQuotas:
    """Model-specific quotas for online inference.

    These quotas are per-model and may vary by region.
    """

    model_id: str
    requests_per_minute: int = 100  # Conservative default RPM
    tokens_per_minute: int = 300_000  # Conservative default TPM
    region: Optional[str] = None

    @classmethod
    def get_default(cls, model_id: str, region: Optional[str] = None) -> "BedrockModelQuotas":
        """Get default quota values for a model.

        Uses static fallback quotas based on known model defaults.
        """
        defaults = _get_static_model_quotas(model_id, region)
        return cls(
            model_id=model_id,
            requests_per_minute=defaults.get("rpm", 100),
            tokens_per_minute=defaults.get("tpm", 300_000),
            region=region,
        )


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
        self._service_quotas_client = None
        self._models_cache: Dict[str, BedrockModelInfo] = {}
        self._converse_supported_cache: Dict[str, bool] = {}
        self._quotas_cache: Optional[BedrockQuotas] = None
        self._model_quotas_cache: Dict[str, BedrockModelQuotas] = {}
        self._quota_name_to_model_cache: Optional[Dict[str, str]] = None
        self._quota_code_cache: Dict[str, Dict[str, str]] = {}  # Cache for quota codes: {model_id: {'rpm_code': 'L-XXX', 'tpm_code': 'L-YYY'}}

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

    @property
    def service_quotas_client(self):
        """Lazily initialize the Service Quotas client."""
        if self._service_quotas_client is None:
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
            self._service_quotas_client = session.client(
                "service-quotas",
                region_name=self.region,
            )
        return self._service_quotas_client

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

    def get_batch_quotas(self, refresh: bool = False) -> BedrockQuotas:
        """Get Bedrock batch inference quotas dynamically from Service Quotas API.

        Falls back to static defaults if API call fails.

        Args:
            refresh: Force refresh the cache

        Returns:
            BedrockQuotas with current quota values
        """
        if self._quotas_cache is not None and not refresh:
            return self._quotas_cache

        quotas = BedrockQuotas.get_default()

        try:
            # List all Bedrock quotas
            paginator = self.service_quotas_client.get_paginator("list_service_quotas")

            for page in paginator.paginate(ServiceCode="bedrock"):
                for quota in page.get("Quotas", []):
                    quota_name = quota.get("QuotaName", "").lower()
                    quota_value = quota.get("Value")

                    if quota_value is None:
                        continue

                    # Match batch-related quotas by name patterns
                    if "batch inference" in quota_name or "batch job" in quota_name:
                        if "records per" in quota_name and "job" in quota_name:
                            if "minimum" in quota_name:
                                quotas.min_records_per_batch_job = int(quota_value)
                            elif "input file" in quota_name:
                                quotas.max_records_per_input_file = int(quota_value)
                            else:
                                quotas.max_records_per_batch_job = int(quota_value)
                        elif "file size" in quota_name or "input file size" in quota_name:
                            # Convert to bytes if needed (quota might be in MB or GB)
                            quotas.max_input_file_size_bytes = int(quota_value)
                        elif "job size" in quota_name:
                            quotas.max_batch_job_size_bytes = int(quota_value)
                        elif "concurrent" in quota_name:
                            quotas.max_concurrent_batch_jobs = int(quota_value)

            logger.debug(f"Loaded Bedrock quotas from Service Quotas API: {quotas}")

        except Exception as e:
            logger.debug(f"Failed to retrieve quotas from Service Quotas API: {e}. Using defaults.")

        self._quotas_cache = quotas
        return quotas

    def get_quota_value(self, quota_code: str) -> Optional[float]:
        """Get a specific quota value by its code.

        Args:
            quota_code: The Service Quotas quota code (e.g., 'L-XXXXXXXX')

        Returns:
            The quota value or None if not found
        """
        try:
            response = self.service_quotas_client.get_service_quota(
                ServiceCode="bedrock",
                QuotaCode=quota_code,
            )
            return response.get("Quota", {}).get("Value")
        except Exception as e:
            logger.debug(f"Failed to get quota {quota_code}: {e}")
            return None

    def get_model_quotas(self, model_id: str, refresh: bool = False) -> BedrockModelQuotas:
        """Get model-specific quotas for online inference.

        Attempts to retrieve RPM/TPM quotas from Service Quotas API with static fallbacks.
        Prioritizes Cross-Region Inference (CRI) quotas over on-demand quotas when available.

        Args:
            model_id: The Bedrock model ID
            refresh: Force refresh the cache

        Returns:
            BedrockModelQuotas with RPM and TPM values
        """
        cache_key = f"{model_id}:{self.region}"
        if cache_key in self._model_quotas_cache and not refresh:
            return self._model_quotas_cache[cache_key]

        # Start with static defaults
        quotas = BedrockModelQuotas.get_default(model_id, self.region)

        try:
            # Get model-friendly name for quota matching
            model_name = self._get_model_friendly_name(model_id)
            if not model_name:
                logger.warning(
                    f"Could not find friendly name mapping for model '{model_id}'. "
                    "Quota lookup will use static defaults. Consider adding a mapping."
                )
                self._model_quotas_cache[cache_key] = quotas
                return quotas

            # Check if we have cached quota codes for this model
            if model_id in self._quota_code_cache and not refresh:
                cached_codes = self._quota_code_cache[model_id]
                try:
                    # Use get_service_quota for faster lookup with cached codes
                    if 'rpm_code' in cached_codes:
                        rpm_response = self.service_quotas_client.get_service_quota(
                            ServiceCode="bedrock",
                            QuotaCode=cached_codes['rpm_code'],
                        )
                        rpm_value = rpm_response.get("Quota", {}).get("Value")
                        if rpm_value is not None:
                            quotas.requests_per_minute = int(rpm_value)

                    if 'tpm_code' in cached_codes:
                        tpm_response = self.service_quotas_client.get_service_quota(
                            ServiceCode="bedrock",
                            QuotaCode=cached_codes['tpm_code'],
                        )
                        tpm_value = tpm_response.get("Quota", {}).get("Value")
                        if tpm_value is not None:
                            quotas.tokens_per_minute = int(tpm_value)

                    logger.debug(
                        f"Loaded quotas for {model_id} from cached codes: "
                        f"RPM={quotas.requests_per_minute}, TPM={quotas.tokens_per_minute}"
                    )
                    self._model_quotas_cache[cache_key] = quotas
                    return quotas
                except Exception as e:
                    logger.debug(f"Failed to use cached quota codes for {model_id}: {e}. Falling back to full listing.")
                    # Remove invalid cached codes
                    del self._quota_code_cache[model_id]

            # Search through quotas for matching model
            paginator = self.service_quotas_client.get_paginator("list_service_quotas")

            rpm_found = False
            tpm_found = False
            rpm_code = None
            tpm_code = None

            # Track both CRI and on-demand quotas
            cri_rpm = None
            cri_tpm = None
            cri_rpm_code = None
            cri_tpm_code = None
            ondemand_rpm = None
            ondemand_tpm = None
            ondemand_rpm_code = None
            ondemand_tpm_code = None

            for page in paginator.paginate(ServiceCode="bedrock"):
                for quota in page.get("Quotas", []):
                    quota_name = quota.get("QuotaName", "")
                    quota_value = quota.get("Value")
                    quota_code = quota.get("QuotaCode", "")

                    if quota_value is None:
                        continue

                    quota_name_lower = quota_name.lower()

                    # Check if quota matches this model
                    # Quota names look like:
                    # "On-demand model inference requests per minute for Anthropic Claude 3 Haiku"
                    # "On-demand model inference tokens per minute for Anthropic Claude 3 Haiku"
                    # "Cross-region model inference requests per minute for Anthropic Claude 3 Haiku"
                    # "Cross-region model inference tokens per minute for Anthropic Claude 3 Haiku"
                    if model_name.lower() in quota_name_lower:
                        is_cri = "cross-region" in quota_name_lower
                        is_ondemand = "on-demand" in quota_name_lower

                        if "requests per minute" in quota_name_lower:
                            if is_cri:
                                cri_rpm = int(quota_value)
                                cri_rpm_code = quota_code
                            elif is_ondemand:
                                ondemand_rpm = int(quota_value)
                                ondemand_rpm_code = quota_code
                        elif "tokens per minute" in quota_name_lower:
                            if is_cri:
                                cri_tpm = int(quota_value)
                                cri_tpm_code = quota_code
                            elif is_ondemand:
                                ondemand_tpm = int(quota_value)
                                ondemand_tpm_code = quota_code

                    # Check if we've found both CRI quotas (preferred) or both on-demand quotas
                    if cri_rpm is not None and cri_tpm is not None:
                        rpm_found = True
                        tpm_found = True
                        break

                if cri_rpm is not None and cri_tpm is not None:
                    break

            # Prioritize CRI quotas over on-demand quotas (CRI typically has higher limits)
            if cri_rpm is not None:
                quotas.requests_per_minute = cri_rpm
                rpm_code = cri_rpm_code
                rpm_found = True
                logger.debug(f"Using cross-region RPM quota for {model_id}: {cri_rpm}")
            elif ondemand_rpm is not None:
                quotas.requests_per_minute = ondemand_rpm
                rpm_code = ondemand_rpm_code
                rpm_found = True

            if cri_tpm is not None:
                quotas.tokens_per_minute = cri_tpm
                tpm_code = cri_tpm_code
                tpm_found = True
                logger.debug(f"Using cross-region TPM quota for {model_id}: {cri_tpm}")
            elif ondemand_tpm is not None:
                quotas.tokens_per_minute = ondemand_tpm
                tpm_code = ondemand_tpm_code
                tpm_found = True

            # Cache the quota codes for faster future lookups
            if rpm_code or tpm_code:
                self._quota_code_cache[model_id] = {}
                if rpm_code:
                    self._quota_code_cache[model_id]['rpm_code'] = rpm_code
                if tpm_code:
                    self._quota_code_cache[model_id]['tpm_code'] = tpm_code

            if rpm_found or tpm_found:
                logger.debug(
                    f"Loaded quotas for {model_id} from Service Quotas API: "
                    f"RPM={quotas.requests_per_minute}, TPM={quotas.tokens_per_minute}"
                )
            else:
                logger.warning(
                    f"Model '{model_id}' (friendly name: '{model_name}') did not match any quota in Service Quotas API. "
                    "Using static defaults. This may indicate a missing quota mapping or a new model."
                )

        except Exception as e:
            logger.debug(f"Failed to retrieve model quotas from Service Quotas API: {e}. Using defaults.")

        self._model_quotas_cache[cache_key] = quotas
        return quotas

    def _get_model_friendly_name(self, model_id: str) -> Optional[str]:
        """Convert model ID to friendly name for quota matching.

        Args:
            model_id: The Bedrock model ID (e.g., 'anthropic.claude-3-haiku-20240307-v1:0')

        Returns:
            Friendly name for quota matching (e.g., 'Anthropic Claude 3 Haiku')
        """
        # Map common model ID patterns to quota-friendly names
        model_name_mappings = {
            # Anthropic Claude
            "anthropic.claude-3-haiku": "Anthropic Claude 3 Haiku",
            "anthropic.claude-3-sonnet": "Anthropic Claude 3 Sonnet",
            "anthropic.claude-3-opus": "Anthropic Claude 3 Opus",
            "anthropic.claude-3-5-sonnet-20240620": "Anthropic Claude 3.5 Sonnet",
            "anthropic.claude-3-5-sonnet-20241022": "Anthropic Claude 3.5 Sonnet V2",
            "anthropic.claude-3-5-haiku": "Anthropic Claude 3.5 Haiku",
            "anthropic.claude-3-7-sonnet": "Anthropic Claude 3.7 Sonnet",
            "anthropic.claude-instant": "Anthropic Claude Instant",
            "anthropic.claude-v2": "Anthropic Claude V2",
            # Amazon Nova
            "amazon.nova-micro": "Amazon Nova Micro",
            "amazon.nova-lite": "Amazon Nova Lite",
            "amazon.nova-pro": "Amazon Nova Pro",
            "amazon.nova-canvas": "Amazon Nova Canvas",
            "amazon.nova-premier": "Amazon Nova Premier",
            # Amazon Titan
            "amazon.titan-text-express": "Amazon Titan Text Express",
            "amazon.titan-text-lite": "Amazon Titan Text Lite",
            "amazon.titan-text-premier": "Amazon Titan Text Premier",
            "amazon.titan-embed-text-v1": "Amazon Titan Text Embeddings",
            "amazon.titan-embed-text-v2": "Amazon Titan Text Embeddings V2",
            "amazon.titan-embed-image": "Amazon Titan Multimodal Embeddings",
            "amazon.titan-image-generator-v1": "Amazon Titan Image Generator G1",
            "amazon.titan-image-generator-v2": "Amazon Titan Image Generator G1 V2",
            # Meta Llama
            "meta.llama2-13b-chat": "Meta Llama 2 Chat 13B",
            "meta.llama2-70b-chat": "Meta Llama 2 Chat 70B",
            "meta.llama3-8b-instruct": "Meta Llama 3 8B Instruct",
            "meta.llama3-70b-instruct": "Meta Llama 3 70B Instruct",
            "meta.llama3-1-8b-instruct": "Meta Llama 3.1 8B Instruct",
            "meta.llama3-1-70b-instruct": "Meta Llama 3.1 70B Instruct",
            "meta.llama3-1-405b-instruct": "Meta Llama 3.1 405B Instruct",
            "meta.llama3-2-1b-instruct": "Meta Llama 3.2 1B Instruct",
            "meta.llama3-2-3b-instruct": "Meta Llama 3.2 3B Instruct",
            "meta.llama3-2-11b-instruct": "Meta Llama 3.2 11B Instruct",
            "meta.llama3-2-90b-instruct": "Meta Llama 3.2 90B Instruct",
            "meta.llama3-3-70b-instruct": "Meta Llama 3.3 70B Instruct",
            # Mistral
            "mistral.mistral-7b-instruct": "Mistral 7B Instruct",
            "mistral.mixtral-8x7b-instruct": "Mistral Mixtral 8x7b Instruct",
            "mistral.mistral-large-2402": "Mistral Large",
            "mistral.mistral-large-2407": "Mistral Large 2407",
            "mistral.mistral-small-2402": "Mistral AI Mistral Small",
            # Cohere
            "cohere.command-r-v1": "Cohere Command R",
            "cohere.command-r-plus": "Cohere Command R Plus",
            "cohere.command-text": "Cohere Command",
            "cohere.command-light": "Cohere Command Light",
            "cohere.embed-english": "Cohere Embed English",
            "cohere.embed-multilingual": "Cohere Embed Multilingual",
            # AI21
            "ai21.jamba-instruct": "AI21 Labs Jamba Instruct",
            "ai21.jamba-1-5-large": "AI21 Labs Jamba 1.5 Large",
            "ai21.jamba-1-5-mini": "AI21 Labs Jamba 1.5 Mini",
            "ai21.j2-ultra": "AI21 Labs Jurassic-2 Ultra",
            "ai21.j2-mid": "AI21 Labs Jurassic-2 Mid",
        }

        # Try exact prefix match first
        for pattern, friendly_name in model_name_mappings.items():
            if model_id.startswith(pattern):
                return friendly_name

        # Try partial match
        for pattern, friendly_name in model_name_mappings.items():
            if pattern in model_id:
                return friendly_name

        # Dynamic fallback: try to get model name from Bedrock API
        model_info = self.get_model_info(model_id)
        if model_info and model_info.model_name:
            # The model_name from API is often already in a format suitable for quota matching
            # e.g., "Claude 3 Haiku" from Anthropic
            logger.debug(f"Using dynamic model name for {model_id}: {model_info.model_name}")
            # Construct a friendly name by combining provider and model name
            if model_info.provider and model_info.model_name:
                # Avoid duplication if provider is already in model_name
                if model_info.provider.lower() not in model_info.model_name.lower():
                    return f"{model_info.provider} {model_info.model_name}"
                return model_info.model_name
            return model_info.model_name

        return None


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

# Static model quotas for online inference (RPM/TPM) with region overrides
# Format: model_pattern -> {default: {rpm, tpm}, region_overrides: {region: {rpm, tpm}}}
_STATIC_MODEL_QUOTAS: Dict[str, Dict] = {
    # Anthropic Claude models
    "anthropic.claude-3-haiku": {
        "default": {"rpm": 400, "tpm": 300_000},
        "regions": {
            "us-east-1": {"rpm": 1000, "tpm": 2_000_000},
            "us-west-2": {"rpm": 1000, "tpm": 2_000_000},
            "ap-northeast-1": {"rpm": 200, "tpm": 200_000},
            "ap-southeast-1": {"rpm": 200, "tpm": 200_000},
        },
    },
    "anthropic.claude-3-sonnet": {
        "default": {"rpm": 100, "tpm": 200_000},
        "regions": {
            "us-east-1": {"rpm": 500, "tpm": 1_000_000},
            "us-west-2": {"rpm": 500, "tpm": 1_000_000},
        },
    },
    "anthropic.claude-3-opus": {
        "default": {"rpm": 50, "tpm": 400_000},
    },
    "anthropic.claude-3-5-sonnet": {
        "default": {"rpm": 50, "tpm": 400_000},
        "regions": {
            "us-west-2": {"rpm": 250, "tpm": 2_000_000},
            "us-east-1": {"rpm": 50, "tpm": 400_000},
            "us-east-2": {"rpm": 50, "tpm": 400_000},
        },
    },
    "anthropic.claude-3-5-haiku": {
        "default": {"rpm": 1000, "tpm": 2_000_000},
        "regions": {
            "us-west-1": {"rpm": 400, "tpm": 300_000},
        },
    },
    "anthropic.claude-3-7-sonnet": {
        "default": {"rpm": 125, "tpm": 500_000},
    },
    "anthropic.claude-instant": {
        "default": {"rpm": 400, "tpm": 300_000},
        "regions": {
            "us-east-1": {"rpm": 1000, "tpm": 1_000_000},
            "us-west-2": {"rpm": 1000, "tpm": 1_000_000},
        },
    },
    # Amazon Nova models
    "amazon.nova-micro": {
        "default": {"rpm": 200, "tpm": 200_000},
        "regions": {
            "us-east-1": {"rpm": 2000, "tpm": 4_000_000},
            "eu-west-2": {"rpm": 2000, "tpm": 4_000_000},
        },
    },
    "amazon.nova-lite": {
        "default": {"rpm": 200, "tpm": 200_000},
        "regions": {
            "us-east-1": {"rpm": 2000, "tpm": 4_000_000},
            "eu-west-2": {"rpm": 2000, "tpm": 4_000_000},
        },
    },
    "amazon.nova-pro": {
        "default": {"rpm": 250, "tpm": 1_000_000},
    },
    "amazon.nova-canvas": {
        "default": {"rpm": 100, "tpm": 100_000},
    },
    # Amazon Titan models
    "amazon.titan-text-express": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "amazon.titan-text-lite": {
        "default": {"rpm": 800, "tpm": 300_000},
    },
    "amazon.titan-text-premier": {
        "default": {"rpm": 100, "tpm": 300_000},
    },
    "amazon.titan-embed": {
        "default": {"rpm": 2000, "tpm": 300_000},
    },
    # Meta Llama models
    "meta.llama3-8b-instruct": {
        "default": {"rpm": 800, "tpm": 300_000},
    },
    "meta.llama3-70b-instruct": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "meta.llama3-1-8b-instruct": {
        "default": {"rpm": 800, "tpm": 300_000},
    },
    "meta.llama3-1-70b-instruct": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "meta.llama3-1-405b-instruct": {
        "default": {"rpm": 200, "tpm": 300_000},
    },
    "meta.llama3-2": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "meta.llama3-3-70b-instruct": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "meta.llama2": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    # Mistral models
    "mistral.mistral-7b-instruct": {
        "default": {"rpm": 800, "tpm": 300_000},
    },
    "mistral.mixtral-8x7b-instruct": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "mistral.mistral-large": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "mistral.mistral-small": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    # Cohere models
    "cohere.command-r": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "cohere.command-text": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
    "cohere.command-light": {
        "default": {"rpm": 800, "tpm": 300_000},
    },
    "cohere.embed": {
        "default": {"rpm": 2000, "tpm": 300_000},
    },
    # AI21 models
    "ai21.jamba": {
        "default": {"rpm": 100, "tpm": 300_000},
    },
    "ai21.j2": {
        "default": {"rpm": 400, "tpm": 300_000},
    },
}


def _get_static_model_quotas(model_id: str, region: Optional[str] = None) -> Dict[str, int]:
    """Get static quota defaults for a model.

    Args:
        model_id: The Bedrock model ID
        region: AWS region for region-specific overrides

    Returns:
        Dict with 'rpm' and 'tpm' keys
    """
    # Try to find matching quota pattern
    for pattern, quota_info in _STATIC_MODEL_QUOTAS.items():
        if model_id.startswith(pattern) or pattern in model_id:
            default = quota_info.get("default", {"rpm": 100, "tpm": 300_000})

            # Check for region-specific overrides
            if region and "regions" in quota_info:
                region_override = quota_info["regions"].get(region)
                if region_override:
                    return {**default, **region_override}

            return default

    # Conservative defaults for unknown models
    return {"rpm": 100, "tpm": 300_000}


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


def get_batch_quotas() -> BedrockQuotas:
    """Get Bedrock batch inference quotas.

    Attempts to retrieve from Service Quotas API with static fallbacks.

    Returns:
        BedrockQuotas with current quota values
    """
    try:
        manager = get_availability_manager()
        return manager.get_batch_quotas()
    except Exception:
        return BedrockQuotas.get_default()


def get_model_quotas(model_id: str, region: Optional[str] = None) -> BedrockModelQuotas:
    """Get model-specific quotas for online inference.

    Attempts to retrieve RPM/TPM quotas from Service Quotas API with static fallbacks.

    Args:
        model_id: The Bedrock model ID
        region: AWS region (uses default if not specified)

    Returns:
        BedrockModelQuotas with RPM and TPM values
    """
    try:
        manager = get_availability_manager(region=region)
        return manager.get_model_quotas(model_id)
    except Exception:
        return BedrockModelQuotas.get_default(model_id, region)
