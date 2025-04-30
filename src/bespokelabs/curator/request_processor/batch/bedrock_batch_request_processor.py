"""AWS Bedrock request processor for batch inference.

This module provides a request processor that interfaces with AWS Bedrock
for batch inference requests.
"""

import asyncio
import datetime
import json
import os
import random
import tempfile
import time
import typing as t
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import boto3
from botocore.exceptions import ClientError

from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.batch.base_batch_request_processor import BaseBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchStatus, GenericBatchRequestCounts
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


class ErrorType(Enum):
    """Enum for different types of errors that can occur during batch processing."""

    # Transient errors that can be retried
    TRANSIENT = "transient"  # Temporary issues that might resolve with retries
    THROTTLING = "throttling"  # Rate limiting or quota issues
    NETWORK = "network"  # Network connectivity issues

    # Permanent errors that cannot be resolved with retries
    PERMISSION = "permission"  # Access denied or insufficient permissions
    RESOURCE_NOT_FOUND = "resource_not_found"  # Requested resource doesn't exist
    VALIDATION = "validation"  # Invalid input parameters
    CONFIGURATION = "configuration"  # Misconfiguration issues

    # Other error types
    UNKNOWN = "unknown"  # Unclassified errors
    TIMEOUT = "timeout"  # Operation timed out


class BedrockBatchRequestProcessor(BaseBatchRequestProcessor):
    """Request processor for AWS Bedrock batch inference.

    This class handles submitting batch inference jobs to AWS Bedrock,
    monitoring job status, and retrieving results.

    Args:
        config: Configuration for the batch request processor
        region_name: AWS region to use for Bedrock API calls
        s3_bucket: S3 bucket to use for input/output data (required for batch)
        s3_prefix: Prefix to use for S3 objects within the bucket
        role_arn: IAM role ARN with necessary permissions for Bedrock batch jobs
        use_inference_profile: Whether to use inference profiles instead of direct model IDs
    """

    # List of models known to support batch inference in AWS Bedrock
    # This list should be updated as AWS adds support for more models
    BATCH_SUPPORTED_MODELS = [
        # Anthropic Claude models
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-7-sonnet-20250219-v1:0",

        # Amazon models
        "amazon.titan-multimodal-embeddings-g1-v1",
        "amazon.titan-text-embeddings-v2",
        "amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
        "amazon.nova-pro-v1:0",

        # Meta Llama models
        "meta.llama3-1-8b-instruct-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "meta.llama3-1-405b-instruct-v1:0",
        "meta.llama3-2-1b-instruct-v1:0",
        "meta.llama3-2-3b-instruct-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
        "meta.llama3-2-90b-instruct-v1:0",
        "meta.llama3-3-70b-instruct-v1:0",

        # Mistral AI models
        "mistral.mistral-small-2402-v1:0",  # Mistral Small (24.02)
        "mistral.mistral-large-2407-v1:0",  # Mistral Large (24.07)
    ]

    # List of cross-region inference profiles
    INFERENCE_PROFILES = [
        # Anthropic Claude models
        "us.anthropic.claude-3-haiku-20240307-v1:0",
        "us.anthropic.claude-3-opus-20240229-v1:0",
        "us.anthropic.claude-3-sonnet-20240229-v1:0",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        # Amazon models
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-micro-v1:0",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.titan-text-express-v1:0",
        "us.amazon.titan-text-premier-v1:0",
        "us.amazon.titan-multimodal-embedding-g1-v1:0",
        "us.amazon.titan-text-embedding-v2:0",
        # Meta Llama models
        "us.meta.llama3-1-8b-instruct-v1:0",
        "us.meta.llama3-1-70b-instruct-v1:0",
        "us.meta.llama3-1-405b-instruct-v1:0",
        "us.meta.llama3-2-1b-instruct-v1:0",
        "us.meta.llama3-2-3b-instruct-v1:0",
        "us.meta.llama3-2-11b-instruct-v1:0",
        "us.meta.llama3-2-90b-instruct-v1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
        # Mistral AI models
        "us.mistral.mistral-small-2402-v1:0",
        "us.mistral.mistral-large-2407-v1:0"
    ]

    def __init__(
        self,
        config: BatchRequestProcessorConfig,
        region_name: str = None,
        s3_bucket: str = None,
        s3_prefix: str = None,
        role_arn: str = None,
        use_inference_profile: bool = False,
    ):
        """Initialize the BedrockBatchRequestProcessor."""
        super().__init__(config)

        # Initialize AWS clients
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.bedrock = boto3.client("bedrock", region_name=self.region_name)
        self.s3 = boto3.client("s3", region_name=self.region_name)
        self._clients = [self.bedrock, self.s3]  # Track clients for cleanup

        # Set model ID and other properties
        self.model_id = config.model
        self.model_provider = self._determine_model_provider()

        # Initialize prompt formatter with default values if not set by parent class
        from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
        if not hasattr(self, 'prompt_formatter'):
            def default_prompt_func(row):
                if isinstance(row, str):
                    return row
                return row.get("prompt", "")

            self.prompt_formatter = PromptFormatter(
                model_name=self.model_id,
                prompt_func=default_prompt_func
            )

        # Initialize other required attributes
        self.total_requests = 1  # Default value, will be updated when processing actual requests
        self.working_dir = tempfile.mkdtemp() if not hasattr(self, 'working_dir') else self.working_dir

        # Initialize viewer client with a dummy implementation if not set
        if not hasattr(self, '_viewer_client') or self._viewer_client is None:
            # Import the actual Client class
            from bespokelabs.curator.client import Client
            self._viewer_client = Client()

        # S3 configuration - critical for batch processing
        self.s3_bucket = s3_bucket or os.environ.get("BEDROCK_BATCH_S3_BUCKET")
        if not self.s3_bucket:
            raise ValueError("S3 bucket must be provided for Bedrock batch processing")

        self.s3_prefix = s3_prefix or os.environ.get("BEDROCK_BATCH_S3_PREFIX", "bedrock-batch")

        # IAM role for batch jobs
        self.role_arn = role_arn or os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        if not self.role_arn:
            raise ValueError("IAM role ARN must be provided for Bedrock batch processing")

        # Batch job tracking
        self.model_id = config.model
        self.use_inference_profile = use_inference_profile

        # If using inference profiles, check if the model ID is already a profile or needs conversion
        if self.use_inference_profile:
            self.model_id = self._get_inference_profile_id(self.model_id)

        # Check if model supports batch inference
        self._validate_batch_support()

        # Determine model provider
        self.model_provider = self._determine_model_provider()

        # Timeout and monitoring configuration
        self.timeout_hours = getattr(config, "timeout_hours", 24)  # Default timeout in hours
        self.monitoring_interval = getattr(config, "monitoring_interval", 300)  # Default: check every 5 minutes
        self.batch_jobs = {}  # Track batch jobs: {batch_id: job_details}

        # Retry configuration
        self.max_retries = getattr(config, "max_retries", 5)
        self.base_delay = getattr(config, "base_delay", 1.0)  # Base delay in seconds
        self.max_delay = getattr(config, "max_delay", 60.0)  # Maximum delay in seconds
        self.jitter_factor = getattr(config, "jitter_factor", 0.1)  # Jitter factor (0.0-1.0)

        # Cost optimization
        self.batch_size_optimization = getattr(config, "batch_size_optimization", True)
        self.optimal_batch_size = getattr(config, "optimal_batch_size", 100)  # Default optimal batch size

        # Initialize semaphore for controlling concurrent operations
        self.semaphore = asyncio.Semaphore(self.max_concurrent_batch_operations)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with resource cleanup."""
        await self.cleanup_resources()
        return False  # Don't suppress exceptions

    async def cleanup_resources(self):
        """Clean up boto3 resources to prevent leaks."""
        logger.debug("Cleaning up boto3 resources")
        # Close all boto3 clients
        for client in self._clients:
            if hasattr(client, 'close') and callable(client.close):
                try:
                    client.close()
                    logger.debug(f"Closed boto3 client: {client.__class__.__name__}")
                except Exception as e:
                    logger.warning(f"Error closing boto3 client: {str(e)}")

    @asynccontextmanager
    async def _retry_with_backoff(self, operation_name: str):
        """Context manager for retrying operations with exponential backoff and jitter.

        Args:
            operation_name: Name of the operation being performed (for logging)

        Yields:
            None

        Raises:
            Exception: If all retries fail
        """
        retries = 0
        while True:
            try:
                yield
                break  # Success, exit the loop
            except Exception as e:
                error_type = self._classify_error(e)

                # Don't retry permanent errors
                if error_type in [ErrorType.PERMISSION, ErrorType.VALIDATION,
                                 ErrorType.CONFIGURATION, ErrorType.RESOURCE_NOT_FOUND]:
                    logger.error(f"{operation_name} failed with permanent error ({error_type.value}): {str(e)}")
                    raise

                retries += 1
                if retries > self.max_retries:
                    logger.error(f"{operation_name} failed after {self.max_retries} retries: {str(e)}")
                    raise

                # Calculate delay with exponential backoff and jitter
                delay = min(self.max_delay, self.base_delay * (2 ** (retries - 1)))
                jitter = delay * self.jitter_factor * random.uniform(-1, 1)
                delay = max(0, delay + jitter)

                # For throttling errors, use a longer delay
                if error_type == ErrorType.THROTTLING:
                    delay = delay * 2

                logger.warning(f"{operation_name} failed (retry {retries}/{self.max_retries}): {str(e)}")
                logger.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an error to determine appropriate handling strategy.

        Args:
            error: The exception to classify

        Returns:
            ErrorType: The classified error type
        """
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')

            # Throttling errors
            if error_code in ['ThrottlingException', 'RequestLimitExceeded', 'TooManyRequestsException',
                             'Throttling', 'RequestThrottled', 'RequestThrottledException']:
                return ErrorType.THROTTLING

            # Permission errors
            elif error_code in ['AccessDenied', 'UnauthorizedOperation', 'AuthorizationError',
                               'AuthFailure', 'InvalidAccessKeyId', 'SignatureDoesNotMatch']:
                return ErrorType.PERMISSION

            # Resource not found errors
            elif error_code in ['ResourceNotFoundException', 'NoSuchEntity', 'NoSuchBucket',
                               'NoSuchKey', 'NoSuchJob', 'NoSuchModel']:
                return ErrorType.RESOURCE_NOT_FOUND

            # Validation errors
            elif error_code in ['ValidationError', 'InvalidParameterValue', 'InvalidInput',
                               'MalformedQueryString', 'InvalidArgument']:
                return ErrorType.VALIDATION

            # Network errors
            elif error_code in ['RequestTimeout', 'ConnectionError', 'ConnectTimeoutError',
                               'ReadTimeoutError']:
                return ErrorType.NETWORK

            # Default to transient for other AWS errors
            return ErrorType.TRANSIENT

        # Classify based on exception type
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ErrorType.TIMEOUT
        elif isinstance(error, (ConnectionError, ConnectionRefusedError, ConnectionResetError)):
            return ErrorType.NETWORK
        elif isinstance(error, (PermissionError, OSError)) and "Permission denied" in str(error):
            return ErrorType.PERMISSION
        elif isinstance(error, FileNotFoundError):
            return ErrorType.RESOURCE_NOT_FOUND
        elif isinstance(error, ValueError) and "configuration" in str(error).lower():
            return ErrorType.CONFIGURATION

        # Default to unknown
        return ErrorType.UNKNOWN

    async def _paginated_s3_list(self, bucket: str, prefix: str) -> t.List[dict]:
        """List S3 objects with pagination support.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix

        Returns:
            List of S3 object metadata dictionaries
        """
        all_objects = []
        continuation_token = None

        while True:
            # Prepare parameters for list_objects_v2
            params = {
                'Bucket': bucket,
                'Prefix': prefix
            }

            if continuation_token:
                params['ContinuationToken'] = continuation_token

            # Make the API call with retry logic
            async with self._retry_with_backoff("S3 list_objects_v2"):
                response = await asyncio.to_thread(
                    self.s3.list_objects_v2,
                    **params
                )

            # Add objects to the result list
            all_objects.extend(response.get('Contents', []))

            # Check if there are more objects to fetch
            if not response.get('IsTruncated'):
                break

            continuation_token = response.get('NextContinuationToken')

        return all_objects

    def _get_inference_profile_id(self, model_id: str) -> str:
        """Convert a standard model ID to an inference profile ID if possible.

        Args:
            model_id: The model ID to convert

        Returns:
            The inference profile ID if available, otherwise the original model ID
        """
        # If already an inference profile, return as is
        if model_id.startswith("us."):
            if model_id in self.INFERENCE_PROFILES:
                logger.info(f"Using inference profile: {model_id}")
                return model_id
            else:
                logger.warning(f"Unknown inference profile: {model_id}, will use as provided")
                return model_id

        # Try to find a matching inference profile
        base_model_name = model_id.split(":")[0] if ":" in model_id else model_id

        for profile in self.INFERENCE_PROFILES:
            # Check if the profile contains the base model name
            if base_model_name in profile:
                logger.info(f"Converting {model_id} to inference profile: {profile}")
                return profile

        # No matching profile found, use the model ID as is
        logger.warning(f"No matching inference profile found for {model_id}, using standard model ID")
        return model_id

    def _validate_batch_support(self) -> None:
        """Validate that the model supports batch inference.

        Raises:
            ValueError: If the model does not support batch inference
        """
        # Skip validation for inference profiles as they're designed for distribution
        if self.model_id.startswith("us."):
            return

        # Check if the model is in the list of supported models
        model_base = self.model_id.split(":")[0] if ":" in self.model_id else self.model_id

        if not any(model_base in supported_model for supported_model in self.BATCH_SUPPORTED_MODELS):
            # Try checking with the AWS Bedrock API to see if the model is batch-enabled
            try:
                response = self.bedrock.get_foundation_model(
                    modelIdentifier=self.model_id
                )
                if not response.get("inferenceTypesSupported") or "ON_DEMAND" not in response.get("inferenceTypesSupported"):
                    logger.warning(f"Model {self.model_id} may not support batch inference according to AWS API")
            except Exception as e:
                logger.warning(f"Could not verify batch support via AWS API: {str(e)}")

            logger.warning(f"Model {self.model_id} is not in the list of known batch-supported models")
            logger.warning("This may still work if AWS has recently added batch support for this model")

    def _get_account_id(self) -> str:
        """Get the AWS account ID.

        Returns:
            str: The AWS account ID
        """
        try:
            # Use STS to get the account ID
            sts = boto3.client('sts', region_name=self.region_name)
            return sts.get_caller_identity()["Account"]
        except Exception as e:
            logger.warning(f"Failed to get AWS account ID: {str(e)}")
            return "000000000000"  # Default placeholder

    def _determine_model_provider(self) -> str:
        """Determine the model provider based on the model ID.

        Returns:
            The model provider as a string
        """
        model_id_lower = self.model_id.lower()

        # For inference profiles, strip the 'us.' prefix for provider detection
        if model_id_lower.startswith("us."):
            model_id_lower = model_id_lower[3:]

        if any(p in model_id_lower for p in ["claude", "anthropic"]):
            return "anthropic"
        elif any(p in model_id_lower for p in ["titan", "amazon", "amazon-titan", "nova"]):
            return "amazon"
        elif any(p in model_id_lower for p in ["llama", "meta"]):
            return "meta"
        elif any(p in model_id_lower for p in ["cohere"]):
            return "cohere"
        elif any(p in model_id_lower for p in ["ai21", "jurassic", "jamba"]):
            return "ai21"
        elif any(p in model_id_lower for p in ["mistral"]):
            return "mistral"
        else:
            logger.warning(f"Unknown model provider for model: {self.model_id}, defaulting to amazon")
            return "amazon"

    @property
    def backend(self) -> str:
        """Return the backend name.

        Returns:
            The backend name as a string
        """
        return "bedrock"

    @property
    def compatible_provider(self) -> str:
        """Return the compatible provider name.

        Returns:
            The compatible provider name as a string
        """
        return "bedrock"

    @property
    def max_requests_per_batch(self) -> int:
        """Return the maximum number of requests allowed per batch.

        Returns:
            Maximum requests per batch
        """
        # AWS Bedrock has a limit of 10,000 records per batch job
        return 10000

    @property
    def max_bytes_per_batch(self) -> int:
        """Maximum size in bytes allowed for a single batch.

        Returns:
            Maximum batch size in bytes
        """
        # AWS Bedrock has a limit of 100 MB per batch job
        return 100 * 1024 * 1024  # 100 MB

    @property
    def max_concurrent_batch_operations(self) -> int:
        """Maximum number of concurrent batch operations.

        Returns:
            Maximum concurrent operations
        """
        # Limit concurrent operations to avoid rate limiting
        return 5

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Convert generic request to API-specific format.

        Args:
            generic_request: Standardized request object.

        Returns:
            dict: API-specific request dictionary formatted for AWS Bedrock.
        """
        # Instead of using the async method, implement the logic directly here
        # Format the request based on the model provider
        model_input = {}

        if self.model_provider == "anthropic":
            # Claude models
            messages = []

            # Handle messages field
            if hasattr(generic_request, 'messages') and generic_request.messages:
                for message in generic_request.messages:
                    if isinstance(message, dict):
                        # Convert to Anthropic format if needed
                        role = message.get("role", "user")
                        content = message.get("content", "")

                        # Handle content based on type
                        if isinstance(content, list):
                            # Already in Anthropic format
                            formatted_content = content
                        elif isinstance(content, str):
                            # Convert string to Anthropic format
                            formatted_content = [{"type": "text", "text": content}]
                        else:
                            # Convert other types to string
                            formatted_content = [{"type": "text", "text": str(content)}]

                        messages.append({
                            "role": role,
                            "content": formatted_content
                        })
            # Fallback to prompt field if available
            elif hasattr(generic_request, 'prompt'):
                if isinstance(generic_request.prompt, list):
                    for message in generic_request.prompt:
                        if isinstance(message, dict):
                            messages.append(message)
                        else:
                            messages.append({
                                "role": "user",
                                "content": [{"type": "text", "text": message}]
                            })
                else:
                    messages = [{
                        "role": "user",
                        "content": [{"type": "text", "text": generic_request.prompt}]
                    }]

            model_input = {
                "messages": messages,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": generic_request.generation_params.get("max_tokens", 1000)
            }

            # Add optional parameters
            for param_name in ["temperature", "top_p", "top_k", "stop_sequences"]:
                if param_name in generic_request.generation_params:
                    model_input[param_name] = generic_request.generation_params[param_name]

        # Format in AWS Bedrock batch format
        # Use task_id if available, otherwise use original_row_idx or a default
        record_id = getattr(generic_request, 'task_id', None)
        if record_id is None:
            record_id = getattr(generic_request, 'original_row_idx', 0)

        return {
            "recordId": f"{record_id}",
            "modelInput": model_input
        }

    async def format_request_for_batch(self, generic_request: GenericRequest) -> dict:
        """Format a generic request for AWS Bedrock batch processing.

        Args:
            generic_request: The generic request to format

        Returns:
            Dict containing the formatted request for batch processing
        """
        # Format the request based on the model provider
        model_input = {}

        if self.model_provider == "anthropic":
            # Claude models
            messages = []

            # Handle messages field
            if hasattr(generic_request, 'messages') and generic_request.messages:
                for message in generic_request.messages:
                    if isinstance(message, dict):
                        # Convert to Anthropic format if needed
                        role = message.get("role", "user")
                        content = message.get("content", "")

                        # Handle content based on type
                        if isinstance(content, list):
                            # Already in Anthropic format
                            formatted_content = content
                        elif isinstance(content, str):
                            # Convert string to Anthropic format
                            formatted_content = [{"type": "text", "text": content}]
                        else:
                            # Convert other types to string
                            formatted_content = [{"type": "text", "text": str(content)}]

                        messages.append({
                            "role": role,
                            "content": formatted_content
                        })
            # Fallback to prompt field if available
            elif hasattr(generic_request, 'prompt'):
                if isinstance(generic_request.prompt, list):
                    for message in generic_request.prompt:
                        if isinstance(message, dict):
                            messages.append(message)
                        else:
                            messages.append({
                                "role": "user",
                                "content": [{"type": "text", "text": message}]
                            })
                else:
                    messages = [{
                        "role": "user",
                        "content": [{"type": "text", "text": generic_request.prompt}]
                    }]

            model_input = {
                "messages": messages,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": generic_request.generation_params.get("max_tokens", 1000)
            }

            # Add optional parameters
            for param_name in ["temperature", "top_p", "top_k", "stop_sequences"]:
                if param_name in generic_request.generation_params:
                    model_input[param_name] = generic_request.generation_params[param_name]

        elif self.model_provider == "amazon":
            # Amazon Titan models
            if "embed" in self.model_id.lower():
                # For embedding models, extract text from messages or use prompt
                if hasattr(generic_request, 'messages') and generic_request.messages:
                    # Extract text from the last user message
                    text = ""
                    for message in reversed(generic_request.messages):
                        if message.get("role", "") == "user":
                            content = message.get("content", "")
                            if isinstance(content, str):
                                text = content
                                break
                            elif isinstance(content, list):
                                # Extract text from content items
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        text = item.get("text", "")
                                        break
                                if text:
                                    break

                    model_input = {
                        "inputText": text
                    }
                elif hasattr(generic_request, 'prompt'):
                    model_input = {
                        "inputText": generic_request.prompt if isinstance(generic_request.prompt, str) else str(generic_request.prompt)
                    }
                else:
                    model_input = {
                        "inputText": ""
                    }
            else:
                # Text generation config
                text_gen_config = {}
                for param_name, bedrock_name in [
                    ("max_tokens", "maxTokenCount"),
                    ("temperature", "temperature"),
                    ("top_p", "topP"),
                    ("stop_sequences", "stopSequences")
                ]:
                    if param_name in generic_request.generation_params:
                        text_gen_config[bedrock_name] = generic_request.generation_params[param_name]

                # Extract text from messages or use prompt
                if hasattr(generic_request, 'messages') and generic_request.messages:
                    # Combine all user messages
                    text = ""
                    for message in generic_request.messages:
                        if message.get("role", "") == "user":
                            content = message.get("content", "")
                            if isinstance(content, str):
                                text += content + "\n"
                            elif isinstance(content, list):
                                # Extract text from content items
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        text += item.get("text", "") + "\n"

                    model_input = {
                        "inputText": text.strip(),
                        "textGenerationConfig": text_gen_config
                    }
                elif hasattr(generic_request, 'prompt'):
                    model_input = {
                        "inputText": generic_request.prompt if isinstance(generic_request.prompt, str) else str(generic_request.prompt),
                        "textGenerationConfig": text_gen_config
                    }
                else:
                    model_input = {
                        "inputText": "",
                        "textGenerationConfig": text_gen_config
                    }

        elif self.model_provider == "meta":
            # Meta Llama models
            messages = []

            # Handle messages field
            if hasattr(generic_request, 'messages') and generic_request.messages:
                for message in generic_request.messages:
                    if isinstance(message, dict):
                        role = message.get("role", "user")
                        content = message.get("content", "")

                        # Handle content based on type
                        if isinstance(content, list):
                            # Extract text from content items
                            text = ""
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text += item.get("text", "") + " "
                            messages.append({
                                "role": role,
                                "content": text.strip()
                            })
                        else:
                            messages.append({
                                "role": role,
                                "content": str(content)
                            })
            # Fallback to prompt field if available
            elif hasattr(generic_request, 'prompt'):
                if isinstance(generic_request.prompt, list):
                    for message in generic_request.prompt:
                        if isinstance(message, dict):
                            messages.append(message)
                        else:
                            messages.append({
                                "role": "user",
                                "content": str(message)
                            })
                else:
                    messages = [{
                        "role": "user",
                        "content": str(generic_request.prompt)
                    }]

            model_input = {
                "messages": messages
            }

            # Add optional parameters
            for param_name, api_name in [
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("max_tokens", "max_gen_len")
            ]:
                if param_name in generic_request.generation_params:
                    model_input[api_name] = generic_request.generation_params[param_name]

        elif self.model_provider == "cohere":
            # Cohere models
            if "embed" in self.model_id.lower():
                model_input = {
                    "texts": [generic_request.prompt] if isinstance(generic_request.prompt, str) else [str(generic_request.prompt)],
                    "input_type": "search_document",
                    "truncate": "END"
                }
            elif "command-r" in self.model_id.lower():
                # Command R format - handle chat history
                if isinstance(generic_request.prompt, list):
                    chat_history = []
                    for message in generic_request.prompt[:-1]:
                        if isinstance(message, dict):
                            role = "USER" if message.get("role", "").lower() == "user" else "CHATBOT"
                            content = message.get("content", "")
                            chat_history.append({"role": role, "message": content})
                        else:
                            role = "USER" if len(chat_history) % 2 == 0 else "CHATBOT"
                            chat_history.append({"role": role, "message": str(message)})

                    last_message = generic_request.prompt[-1]
                    if isinstance(last_message, dict):
                        query = last_message.get("content", "")
                    else:
                        query = str(last_message)
                else:
                    query = generic_request.prompt
                    chat_history = []

                model_input = {
                    "message": query,
                    "chat_history": chat_history
                }
            else:
                # Command format
                prompt = generic_request.prompt
                if isinstance(prompt, list):
                    prompt = "\n".join([str(p) for p in prompt])

                model_input = {
                    "prompt": prompt
                }

            # Add optional parameters
            param_mapping = {
                "temperature": "temperature",
                "top_p": "p",
                "top_k": "k",
                "max_tokens": "max_tokens",
                "stop_sequences": "stop_sequences"
            }

            for param_name, api_name in param_mapping.items():
                if param_name in generic_request.generation_params:
                    model_input[api_name] = generic_request.generation_params[param_name]

        elif self.model_provider == "ai21":
            # AI21 models
            if "jamba" in self.model_id.lower():
                # Jamba format
                if isinstance(generic_request.prompt, list):
                    messages = []
                    for message in generic_request.prompt:
                        if isinstance(message, dict):
                            messages.append(message)
                        else:
                            messages.append({
                                "role": "user",
                                "content": str(message)
                            })
                else:
                    messages = [{
                        "role": "user",
                        "content": str(generic_request.prompt)
                    }]

                model_input = {
                    "messages": messages
                }
            else:
                # Jurassic format
                prompt = generic_request.prompt
                if isinstance(prompt, list):
                    prompt = "\n".join([str(p) for p in prompt])

                model_input = {
                    "prompt": prompt
                }

                # Add penalty parameters
                for penalty in ["countPenalty", "presencePenalty", "frequencyPenalty"]:
                    if penalty in generic_request.generation_params:
                        model_input[penalty] = {
                            "scale": generic_request.generation_params[penalty]
                        }

            # Add optional parameters
            param_mapping = {
                "temperature": "temperature",
                "top_p": "topP" if "jamba" not in self.model_id.lower() else "top_p",
                "max_tokens": "maxTokens" if "jamba" not in self.model_id.lower() else "max_tokens",
                "stop_sequences": "stopSequences" if "jamba" not in self.model_id.lower() else "stop_sequences"
            }

            for param_name, api_name in param_mapping.items():
                if param_name in generic_request.generation_params:
                    model_input[api_name] = generic_request.generation_params[param_name]

        elif self.model_provider == "mistral":
            # Mistral AI models
            if isinstance(generic_request.prompt, list):
                messages = []
                for message in generic_request.prompt:
                    if isinstance(message, dict):
                        messages.append(message)
                    else:
                        messages.append({
                            "role": "user",
                            "content": str(message)
                        })
            else:
                messages = [{
                    "role": "user",
                    "content": str(generic_request.prompt)
                }]

            model_input = {
                "messages": messages
            }

            # Add optional parameters
            for param_name in ["temperature", "top_p", "max_tokens"]:
                if param_name in generic_request.generation_params:
                    model_input[param_name] = generic_request.generation_params[param_name]
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        # Format in AWS Bedrock batch format
        # Use task_id if available, otherwise use original_row_idx or a default
        record_id = getattr(generic_request, 'task_id', None)
        if record_id is None:
            record_id = getattr(generic_request, 'original_row_idx', 0)

        return {
            "recordId": f"{record_id}",
            "modelInput": model_input
        }

    async def submit_batch(self, requests: list[dict], metadata: dict) -> GenericBatch:
        """Submit a batch of requests to AWS Bedrock.

        Args:
            requests: List of API-specific request dictionaries
            metadata: Metadata to associate with the batch

        Returns:
            GenericBatch: The created batch object

        Raises:
            Exception: If batch submission fails
        """
        async with self.semaphore:
            # Generate a unique batch ID
            batch_id = f"bedrock-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            request_file = metadata.get("request_file")

            # Apply batch size optimization if enabled
            if self.batch_size_optimization and len(requests) > self.optimal_batch_size:
                logger.info(f"Optimizing batch size: {len(requests)} requests will be processed in chunks of {self.optimal_batch_size}")
                # Note: The actual chunking happens at a higher level, this is just for logging

            # Create batch input file in JSONL format
            batch_file_path = Path(self.working_dir) / f"{batch_id}_input.jsonl"

            try:
                # Write requests to batch file
                async with aiofiles.open(batch_file_path, "w") as f:
                    for request in requests:
                        await f.write(json.dumps(request) + "\n")

                logger.info(f"Created batch file with {len(requests)} requests at {batch_file_path}")

                # Upload batch file to S3 with retry logic
                input_s3_key = f"{self.s3_prefix}/input/{batch_id}_input.jsonl"
                async with self._retry_with_backoff(f"Upload batch file to S3 {input_s3_key}"):
                    await asyncio.to_thread(
                        self.s3.upload_file,
                        str(batch_file_path),
                        self.s3_bucket,
                        input_s3_key
                    )

                logger.info(f"Uploaded batch file to s3://{self.s3_bucket}/{input_s3_key}")

                # Set up S3 locations for input and output
                s3_input_uri = f"s3://{self.s3_bucket}/{input_s3_key}"
                s3_output_uri = f"s3://{self.s3_bucket}/{self.s3_prefix}/output/{batch_id}/"

                # Prepare job parameters with cost optimization settings
                job_params = {
                    "jobName": f"curator-batch-{batch_id}",
                    "modelId": self.model_id,
                    "roleArn": self.role_arn,
                    "inputDataConfig": {
                        "s3InputDataConfig": {
                            "s3Uri": s3_input_uri
                        }
                    },
                    "outputDataConfig": {
                        "s3OutputDataConfig": {
                            "s3Uri": s3_output_uri
                        }
                    },
                    "timeoutDurationInHours": self.timeout_hours
                }

                # Add cost optimization parameters if available
                # AWS Bedrock offers 50% cost savings for non-urgent batch jobs
                # by setting lower priority for the job
                if hasattr(self.config, "use_cost_optimization") and self.config.use_cost_optimization:
                    # Note: This is a placeholder for when AWS adds explicit cost optimization parameters
                    # Currently, the cost savings are automatic for batch jobs
                    logger.info("Using cost optimization for batch job")

                    # Some providers might support additional parameters for cost optimization
                    if self.model_provider == "anthropic" and hasattr(self.config, "low_priority"):
                        job_params["jobPriority"] = "LOW" if self.config.low_priority else "NORMAL"

                # Create batch job in AWS Bedrock with retry logic
                async with self._retry_with_backoff(f"Create batch job for {batch_id}"):
                    job_response = await asyncio.to_thread(
                        self.bedrock.create_model_invocation_job,
                        **job_params
                    )

                # Store job details
                job_id = job_response.get("jobArn").split("/")[-1]

                logger.info(f"Submitted batch job {job_id} for batch {batch_id}")

                # Create a GenericBatch object
                batch = GenericBatch(
                    batch_id=batch_id,
                    id=batch_id,  # Use the same ID for both fields
                    provider_batch_id=job_id,
                    status=GenericBatchStatus.PROCESSING,
                    request_file=request_file,
                    metadata={
                        "job_id": job_id,
                        "job_arn": job_response.get("jobArn"),
                        "model_id": self.model_id,
                        "using_inference_profile": self.use_inference_profile,
                        "s3_input_uri": s3_input_uri,
                        "s3_output_uri": s3_output_uri,
                        "submission_time": datetime.datetime.now().isoformat(),
                        "num_requests": len(requests),
                        "cost_optimized": hasattr(self.config, "use_cost_optimization") and self.config.use_cost_optimization
                    },
                    request_counts=GenericBatchRequestCounts(
                        total=len(requests),
                        succeeded=0,
                        failed=0,
                        raw_request_counts_object={"total": len(requests), "succeeded": 0, "failed": 0}
                    )
                )

                # Store batch job info for later reference
                self.batch_jobs[batch_id] = {
                    "job_id": job_id,
                    "input_s3_key": input_s3_key,
                    "output_s3_uri": s3_output_uri,
                    "batch_file_path": str(batch_file_path),
                    "submission_time": datetime.datetime.now()
                }

                return batch

            except Exception as e:
                error_type = self._classify_error(e)
                logger.error(f"Failed to submit batch: {str(e)} (type: {error_type.value})")
                raise Exception(f"Failed to submit batch: {str(e)}")



    async def fetch_batch_results(self, batch: GenericBatch) -> t.List[GenericResponse]:
        """Fetch results for a completed batch job from AWS Bedrock.

        Args:
            batch: The completed batch to fetch results for

        Returns:
            List of GenericResponse objects

        Raises:
            Exception: If result fetching fails
        """
        if batch.status != GenericBatchStatus.COMPLETE:
            raise ValueError(f"Cannot fetch results for batch {batch.batch_id} with status {batch.status}")

        if not batch.provider_batch_id or "metadata" not in batch or not isinstance(batch.metadata, dict):
            raise ValueError(f"Batch {batch.batch_id} has missing provider ID or metadata")

        # Get output S3 URI from metadata
        output_s3_uri = batch.metadata.get("s3_output_uri")
        if not output_s3_uri:
            raise ValueError(f"Batch {batch.batch_id} has no output S3 URI in metadata")

        try:
            # Parse S3 URI
            parsed_uri = urlparse(output_s3_uri)
            bucket = parsed_uri.netloc
            prefix = parsed_uri.path.lstrip("/")

            # List objects in output location with pagination support
            objects = await self._paginated_s3_list(bucket, prefix)

            # Create a mapping from record ID to response
            responses_by_id = {}

            # Process all output files
            for obj in objects:
                key = obj["Key"]

                # Skip any non-response files (e.g., manifests)
                if not key.endswith(".jsonl.out"):
                    continue

                # Download and process response file
                output_file_path = Path(self.working_dir) / f"{batch.batch_id}_output_{Path(key).name}"

                async with self._retry_with_backoff(f"Download S3 file {key}"):
                    await asyncio.to_thread(
                        self.s3.download_file,
                        bucket,
                        key,
                        str(output_file_path)
                    )

                # Parse response file
                async with aiofiles.open(output_file_path, "r") as f:
                    content = await f.read()
                    for line in content.splitlines():
                        try:
                            response_json = json.loads(line)
                            record_id = response_json.get("recordId")
                            if record_id:
                                responses_by_id[record_id] = response_json
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in response file {key}: {line}")

            # Convert to GenericResponse objects
            responses = []
            for request in batch.requests:
                record_id = str(request.task_id)
                response_json = responses_by_id.get(record_id)

                if not response_json:
                    # Missing response
                    responses.append(GenericResponse(
                        task_id=request.task_id,
                        response="",
                        success=False,
                        finish_reason="missing_response",
                        message=f"No response found for request {record_id}",
                        processing_time=0,
                        token_usage=_TokenUsage(input_tokens=0, output_tokens=0)
                    ))
                    continue

                # Extract model output from response
                model_output = response_json.get("modelOutput", {})

                # Process based on model provider
                text_response, token_usage = self._extract_response_from_batch(model_output, request)

                # Create generic response
                responses.append(GenericResponse(
                    task_id=request.task_id,
                    response=text_response,
                    success=True,
                    finish_reason="stop",
                    message="Success",
                    processing_time=0,  # Batch jobs don't provide per-request timing
                    raw_data=model_output,
                    token_usage=token_usage
                ))

            return responses

        except Exception as e:
            error_type = self._classify_error(e)
            logger.error(f"Error fetching results for batch {batch.batch_id}: {str(e)} (type: {error_type.value})")
            raise Exception(f"Failed to fetch batch results: {str(e)}")

    def _extract_response_from_batch(self, model_output: dict, request: GenericRequest) -> t.Tuple[str, _TokenUsage]:
        """Extract response text and token usage from model output.

        Args:
            model_output: The raw model output from batch processing
            request: The original request for context

        Returns:
            Tuple of (text_response, token_usage)
        """
        # Default values
        text_response = ""
        input_tokens = 0
        output_tokens = 0

        # Extract based on model provider
        if self.model_provider == "anthropic":
            # Claude models
            if "content" in model_output:
                for content_item in model_output.get("content", []):
                    if content_item.get("type") == "text":
                        text_response = content_item.get("text", "")

            # Get token usage
            usage = model_output.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

        elif self.model_provider == "amazon":
            # Amazon Titan models
            if "embed" in self.model_id.lower():
                # Embedding response
                text_response = str(model_output.get("embedding", []))
                input_tokens = model_output.get("inputTextTokenCount", 0)
                output_tokens = 0
            else:
                # Text generation model
                text_response = model_output.get("results", [{}])[0].get("outputText", "")
                input_tokens = model_output.get("inputTextTokenCount", 0)
                output_tokens = model_output.get("results", [{}])[0].get("tokenCount", 0)

        elif self.model_provider == "meta":
            # Meta Llama models
            text_response = model_output.get("generation", "")
            input_tokens = model_output.get("prompt_token_count", 0)
            output_tokens = model_output.get("generation_token_count", 0)

        elif self.model_provider == "cohere":
            # Cohere models
            if "embed" in self.model_id.lower():
                # Embedding response
                text_response = str(model_output.get("embeddings", [[]])[0])
                input_tokens = len(str(request.prompt)) // 4  # Rough estimate
                output_tokens = 0
            elif "command-r" in self.model_id.lower():
                # Command R format
                text_response = model_output.get("text", "")
                token_count = model_output.get("token_count", {})
                input_tokens = token_count.get("prompt_tokens", 0)
                output_tokens = token_count.get("completion_tokens", 0)
            else:
                # Command format
                text_response = model_output.get("generations", [{}])[0].get("text", "")
                # Rough token estimate
                input_tokens = len(str(request.prompt)) // 4
                output_tokens = len(text_response) // 4

        elif self.model_provider == "ai21":
            # AI21 models
            if "jamba" in self.model_id.lower():
                # Jamba format
                text_response = model_output.get("message", {}).get("content", "")
                input_tokens = model_output.get("prompt_tokens", 0)
                output_tokens = model_output.get("completion_tokens", 0)
            else:
                # Jurassic format
                text_response = model_output.get("completions", [{}])[0].get("data", {}).get("text", "")
                # Rough token estimate
                input_tokens = len(str(request.prompt)) // 4
                output_tokens = len(text_response) // 4

        elif self.model_provider == "mistral":
            # Mistral AI models
            choices = model_output.get("choices", [{}])
            if choices:
                message = choices[0].get("message", {})
                text_response = message.get("content", "")

            # Get token usage
            usage = model_output.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        # Create token usage object
        token_usage = _TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        return text_response, token_usage

    async def cleanup_batch(self, batch: GenericBatch) -> None:
        """Clean up resources for a batch job.

        Args:
            batch: The batch to clean up
        """
        # Clean up local files
        if batch.batch_id in self.batch_jobs:
            job_info = self.batch_jobs[batch.batch_id]
            batch_file_path = job_info.get("batch_file_path")

            if batch_file_path and os.path.exists(batch_file_path):
                try:
                    if (batch.status == GenericBatchStatus.COMPLETE and
                        getattr(self.config, "delete_successful_batch_files", False)):
                        os.remove(batch_file_path)
                        logger.info(f"Deleted successful batch input file: {batch_file_path}")
                    elif (batch.status == GenericBatchStatus.FAILED and
                          getattr(self.config, "delete_failed_batch_files", False)):
                        os.remove(batch_file_path)
                        logger.info(f"Deleted failed batch input file: {batch_file_path}")
                except Exception as e:
                    error_type = self._classify_error(e)
                    logger.warning(f"Failed to delete batch file {batch_file_path}: {str(e)} (type: {error_type.value})")

            # Clean up S3 files
            if getattr(self.config, "delete_successful_batch_files", False) and batch.status == GenericBatchStatus.COMPLETE:
                try:
                    # Delete input file from S3
                    input_s3_key = job_info.get("input_s3_key")
                    if input_s3_key:
                        async with self._retry_with_backoff(f"Delete S3 input file {input_s3_key}"):
                            await asyncio.to_thread(
                                self.s3.delete_object,
                                Bucket=self.s3_bucket,
                                Key=input_s3_key
                            )
                        logger.info(f"Deleted S3 input file: s3://{self.s3_bucket}/{input_s3_key}")

                    # Delete output files from S3
                    output_s3_uri = job_info.get("output_s3_uri")
                    if output_s3_uri:
                        parsed_uri = urlparse(output_s3_uri)
                        bucket = parsed_uri.netloc
                        prefix = parsed_uri.path.lstrip("/")

                        # List objects with pagination support
                        objects = await self._paginated_s3_list(bucket, prefix)

                        # Delete objects in batches to improve performance
                        batch_size = 100  # AWS allows up to 1000 objects per delete operation
                        for i in range(0, len(objects), batch_size):
                            batch_objects = objects[i:i+batch_size]

                            # Skip if batch is empty
                            if not batch_objects:
                                continue

                            # Delete objects in this batch
                            for obj in batch_objects:
                                async with self._retry_with_backoff(f"Delete S3 output file {obj['Key']}"):
                                    await asyncio.to_thread(
                                        self.s3.delete_object,
                                        Bucket=self.s3_bucket,
                                        Key=obj["Key"]
                                    )

                        logger.info(f"Deleted S3 output files: {output_s3_uri}")
                except Exception as e:
                    error_type = self._classify_error(e)
                    logger.warning(f"Failed to delete S3 files for batch {batch.batch_id}: {str(e)} (type: {error_type.value})")

            # Remove batch from tracking
            self.batch_jobs.pop(batch.batch_id, None)

    async def monitor_batch_job(self, batch: GenericBatch, timeout_seconds: int = None) -> GenericBatch:
        """Monitor a batch job until completion or timeout.

        Args:
            batch: The batch to monitor
            timeout_seconds: Optional timeout in seconds (overrides self.timeout_hours)

        Returns:
            GenericBatch: Updated batch object with final status

        Raises:
            TimeoutError: If the job exceeds the timeout period
        """
        if not timeout_seconds:
            timeout_seconds = self.timeout_hours * 3600  # Convert hours to seconds

        start_time = time.time()
        check_interval = self.monitoring_interval  # Start with the default interval

        while True:
            # Check if we've exceeded the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                logger.error(f"Batch job {batch.batch_id} timed out after {elapsed_time:.1f} seconds")

                # Try to cancel the job
                try:
                    await self.cancel_batch(batch)
                except Exception as e:
                    logger.warning(f"Failed to cancel timed out batch {batch.batch_id}: {str(e)}")

                # Update batch status
                batch.status = GenericBatchStatus.FAILED
                if not batch.metadata:
                    batch.metadata = {}
                batch.metadata.update({
                    "timeout": True,
                    "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": elapsed_time
                })

                raise TimeoutError(f"Batch job {batch.batch_id} timed out after {elapsed_time:.1f} seconds")

            # Retrieve current status
            updated_batch = await self.retrieve_batch(batch)
            if not updated_batch:
                logger.warning(f"Failed to retrieve status for batch {batch.batch_id}")
                await asyncio.sleep(check_interval)
                continue

            # Update our batch object with the latest info
            batch = updated_batch

            # Check if the job has completed or failed
            if batch.status in [GenericBatchStatus.COMPLETE, GenericBatchStatus.FAILED]:
                logger.info(f"Batch job {batch.batch_id} finished with status: {batch.status}")
                return batch

            # Adjust check interval based on elapsed time for efficiency
            # For longer-running jobs, we don't need to check as frequently
            if elapsed_time > 3600:  # After 1 hour
                check_interval = min(600, self.monitoring_interval * 2)  # Max 10 minutes
            elif elapsed_time > 600:  # After 10 minutes
                check_interval = min(300, self.monitoring_interval * 1.5)  # Max 5 minutes

            # Log progress periodically
            if elapsed_time % 600 < check_interval:  # Log approximately every 10 minutes
                logger.info(f"Batch job {batch.batch_id} still running after {elapsed_time:.1f} seconds")

            # Wait before checking again
            await asyncio.sleep(check_interval)

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieve current status of a submitted batch.

        Args:
            batch: The batch object to check status for.

        Returns:
            GenericBatch: Updated batch object with current status.
            None: If the batch is not found or inaccessible.
        """
        async with self.semaphore:
            if not batch.provider_batch_id:
                logger.warning(f"Batch {batch.batch_id} has no provider batch ID")
                return None

            try:
                # Get the batch status from AWS Bedrock with retry logic
                # Use the full ARN from metadata if available, otherwise fall back to provider_batch_id
                job_identifier = batch.metadata.get("job_arn") if batch.metadata and isinstance(batch.metadata, dict) else batch.provider_batch_id

                # If the job_identifier is not a full ARN but looks like a job ID, construct the ARN
                if job_identifier and not job_identifier.startswith("arn:aws:"):
                    job_identifier = f"arn:aws:bedrock:{self.region_name}:{self._get_account_id()}:model-invocation-job/{job_identifier}"

                async with self._retry_with_backoff(f"Get batch status for {job_identifier}"):
                    response = await asyncio.to_thread(
                        self.bedrock.get_model_invocation_job,
                        jobIdentifier=job_identifier
                    )

                # Map AWS Bedrock status to GenericBatchStatus
                bedrock_status = response.get("status")

                status_mapping = {
                    "Submitted": GenericBatchStatus.PROCESSING,
                    "InProgress": GenericBatchStatus.PROCESSING,
                    "Completed": GenericBatchStatus.COMPLETE,
                    "Failed": GenericBatchStatus.FAILED,
                    "Stopping": GenericBatchStatus.PROCESSING,
                    "Stopped": GenericBatchStatus.FAILED,
                    "Expired": GenericBatchStatus.FAILED
                }

                # Update batch with latest status
                batch.status = status_mapping.get(bedrock_status, GenericBatchStatus.PROCESSING)

                # Update metadata
                if not batch.metadata:
                    batch.metadata = {}

                batch.metadata.update({
                    "bedrock_status": bedrock_status,
                    "last_checked": datetime.datetime.now().isoformat()
                })

                # Add failure reason if available
                if bedrock_status in ["Failed", "Stopped", "Expired"] and "failureMessage" in response:
                    batch.metadata["failure_reason"] = response.get("failureMessage")
                    logger.error(f"Batch {batch.batch_id} failed: {response.get('failureMessage')}")

                return batch

            except Exception as e:
                logger.warning(f"Error retrieving batch {batch.batch_id}: {str(e)}")
                return None

    async def download_batch(self, batch: GenericBatch) -> list[dict] | None:
        """Download results of a completed batch.

        Args:
            batch: The completed batch object to download.

        Returns:
            list[dict] | None: List of response dictionaries if successful, None if download fails.
        """
        async with self.semaphore:
            if batch.status != GenericBatchStatus.COMPLETE:
                logger.warning(f"Cannot download batch {batch.batch_id} with status {batch.status}")
                return None

            if not batch.provider_batch_id or not batch.metadata or not isinstance(batch.metadata, dict):
                logger.warning(f"Batch {batch.batch_id} has missing provider ID or metadata")
                return None

            # Get output S3 URI from metadata
            output_s3_uri = batch.metadata.get("s3_output_uri")
            if not output_s3_uri:
                logger.warning(f"Batch {batch.batch_id} has no output S3 URI in metadata")
                return None

            try:
                # Parse S3 URI
                parsed_uri = urlparse(output_s3_uri)
                bucket = parsed_uri.netloc
                prefix = parsed_uri.path.lstrip("/")

                # List objects in output location with pagination support
                objects = await self._paginated_s3_list(bucket, prefix)

                # Process all output files
                responses = []

                for obj in objects:
                    key = obj["Key"]

                    # Skip any non-response files (e.g., manifests)
                    if not key.endswith(".jsonl.out"):
                        continue

                    # Download and process response file
                    output_file_path = Path(self.working_dir) / f"{batch.batch_id}_output_{Path(key).name}"

                    # Use retry mechanism for downloading
                    async with self._retry_with_backoff(f"Download S3 file {key}"):
                        await asyncio.to_thread(
                            self.s3.download_file,
                            bucket,
                            key,
                            str(output_file_path)
                        )

                    # Parse response file
                    async with aiofiles.open(output_file_path, "r") as f:
                        content = await f.read()
                        for line in content.splitlines():
                            try:
                                response_json = json.loads(line)
                                responses.append(response_json)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in response file {key}: {line}")

                return responses

            except Exception as e:
                error_type = self._classify_error(e)
                logger.error(f"Error downloading batch {batch.batch_id}: {str(e)} (type: {error_type.value})")
                return None

    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parse API-specific response into standardized format.

        Args:
            raw_response: Raw response dictionary from API.
            generic_request: Original generic request object.
            batch: Batch object containing context information.

        Returns:
            GenericResponse: Standardized response object.
        """
        # Extract model output
        model_output = raw_response.get("modelOutput", {})

        # Process based on model provider
        text_response, token_usage = self._extract_response_from_batch(model_output, generic_request)

        # Create generic response
        return GenericResponse(
            task_id=generic_request.task_id,
            response=text_response,
            success=True if text_response else False,
            finish_reason="stop" if text_response else "error",
            message="Success" if text_response else "Failed to extract response",
            processing_time=0,  # Batch jobs don't provide per-request timing
            raw_data=model_output,
            raw_response=raw_response,
            generic_request=generic_request,
            token_usage=token_usage,
            created_at=datetime.datetime.fromisoformat(batch.metadata.get("submission_time", datetime.datetime.now().isoformat())),
            finished_at=datetime.datetime.now()
        )

    def parse_api_specific_batch_object(self, batch: object, request_file: str | None = None) -> GenericBatch:
        """Convert API-specific batch object to generic format.

        Args:
            batch: API-specific batch object (AWS Bedrock job response).
            request_file: Optional path to associated request file.

        Returns:
            GenericBatch: Standardized batch object.
        """
        # For AWS Bedrock, the batch object is a dictionary from the get_model_invocation_job API
        if isinstance(batch, dict):
            # Extract job ID from ARN
            job_arn = batch.get("jobArn", "")
            job_id = job_arn.split("/")[-1] if "/" in job_arn else job_arn

            # Map status
            bedrock_status = batch.get("status")
            status_mapping = {
                "Submitted": GenericBatchStatus.PROCESSING,
                "InProgress": GenericBatchStatus.PROCESSING,
                "Completed": GenericBatchStatus.COMPLETE,
                "Failed": GenericBatchStatus.FAILED,
                "Stopping": GenericBatchStatus.PROCESSING,
                "Stopped": GenericBatchStatus.FAILED,
                "Expired": GenericBatchStatus.FAILED
            }
            status = status_mapping.get(bedrock_status, GenericBatchStatus.PROCESSING)

            # Extract S3 URIs
            input_data_config = batch.get("inputDataConfig", {}).get("s3InputDataConfig", {})
            output_data_config = batch.get("outputDataConfig", {}).get("s3OutputDataConfig", {})

            s3_input_uri = input_data_config.get("s3Uri")
            s3_output_uri = output_data_config.get("s3Uri")

            # Create batch ID from job name or ID
            batch_id = batch.get("jobName", "").replace("curator-batch-", "") or job_id

            # Create metadata
            metadata = {
                "job_id": job_id,
                "job_arn": job_arn,
                "model_id": batch.get("modelId"),
                "s3_input_uri": s3_input_uri,
                "s3_output_uri": s3_output_uri,
                "bedrock_status": bedrock_status,
                "submission_time": batch.get("creationTime", datetime.datetime.now()).isoformat() if hasattr(batch.get("creationTime", None), "isoformat") else datetime.datetime.now().isoformat(),
                "last_checked": datetime.datetime.now().isoformat()
            }

            # Add failure reason if available
            if bedrock_status in ["Failed", "Stopped", "Expired"] and "failureMessage" in batch:
                metadata["failure_reason"] = batch.get("failureMessage")

            return GenericBatch(
                batch_id=batch_id,
                id=batch_id,
                provider_batch_id=job_id,
                status=status,
                request_file=request_file,
                metadata=metadata
            )
        else:
            # If not a dict, assume it's already a GenericBatch
            return batch

    def parse_api_specific_request_counts(self, request_counts: object, request_file: str | None = None) -> GenericBatchRequestCounts:  # noqa: ARG002
        """Convert API-specific request counts to generic format.

        Args:
            request_counts: API-specific request count object.
            request_file: Path to associated request file.

        Returns:
            GenericBatchRequestCounts: Standardized request count object.
        """
        # AWS Bedrock doesn't provide detailed request counts during processing
        # We can only estimate based on the batch status

        # Count the number of requests in the file if available
        total_from_file = 0
        if request_file and os.path.exists(request_file):
            with open(request_file, 'r') as f:
                total_from_file = sum(1 for _ in f)

        if isinstance(request_counts, dict):
            # If we have a job response, extract what we can
            total = request_counts.get("total", total_from_file)
            succeeded = request_counts.get("succeeded", 0)
            failed = request_counts.get("failed", 0)
            pending = total - (succeeded + failed)

            return GenericBatchRequestCounts(
                total=total,
                succeeded=succeeded,
                failed=failed,
                pending=pending,
                raw_request_counts_object=request_counts
            )
        elif isinstance(request_counts, int):
            # If we just have a total count
            total = request_counts or total_from_file
            return GenericBatchRequestCounts(
                total=total,
                succeeded=0,
                failed=0,
                pending=total,
                raw_request_counts_object={"total": total}
            )
        else:
            # Default to using file count or empty counts
            total = total_from_file
            return GenericBatchRequestCounts(
                total=total,
                succeeded=0,
                failed=0,
                pending=total,
                raw_request_counts_object={"total": total}
            )

    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancel a batch job in AWS Bedrock.

        Args:
            batch: The batch to cancel

        Returns:
            GenericBatch: Updated batch object after cancellation

        Raises:
            Exception: If cancellation fails
        """
        async with self.semaphore:
            if not batch.provider_batch_id:
                logger.error(f"Batch {batch.batch_id} has no provider batch ID")
                return batch

            try:
                # Stop the batch job with retry logic
                # Use the full ARN from metadata if available, otherwise fall back to provider_batch_id
                job_identifier = batch.metadata.get("job_arn") if batch.metadata and isinstance(batch.metadata, dict) else batch.provider_batch_id

                # If the job_identifier is not a full ARN but looks like a job ID, construct the ARN
                if job_identifier and not job_identifier.startswith("arn:aws:"):
                    job_identifier = f"arn:aws:bedrock:{self.region_name}:{self._get_account_id()}:model-invocation-job/{job_identifier}"

                async with self._retry_with_backoff(f"Cancel batch job {job_identifier}"):
                    await asyncio.to_thread(
                        self.bedrock.stop_model_invocation_job,
                        jobIdentifier=job_identifier
                    )

                logger.info(f"Cancelled batch job {batch.provider_batch_id} for batch {batch.batch_id}")

                # Update batch status
                batch.status = GenericBatchStatus.FAILED
                batch.metadata = batch.metadata or {}
                batch.metadata.update({
                    "cancelled": True,
                    "cancelled_time": datetime.datetime.now().isoformat()
                })

                # Clean up resources
                await self.cleanup_batch(batch)

                return batch

            except Exception as e:
                error_type = self._classify_error(e)
                logger.error(f"Failed to cancel batch {batch.batch_id}: {str(e)} (type: {error_type.value})")
                raise Exception(f"Failed to cancel batch: {str(e)}")