"""AWS Bedrock request processor for online inference.

This module provides a request processor that interfaces with AWS Bedrock
for real-time inference requests, prioritizing the Converse API for chat models.
"""

import asyncio
import datetime
import json
import os
import re
import typing as t
from io import BytesIO
from pathlib import Path

import aiohttp
import boto3
from botocore.exceptions import ClientError

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.online.base_online_request_processor import (
    APIRequest,
    BaseOnlineRequestProcessor,
)
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


class BedrockOnlineRequestProcessor(BaseOnlineRequestProcessor):
    """Request processor for AWS Bedrock API.

    This class handles making real-time inference requests to the AWS Bedrock API,
    with support for different model providers hosted on Bedrock. It prioritizes
    using the Converse API for chat-based interactions, as it provides better support
    for multi-turn conversations and consistent message formatting.

    Args:
        config: Configuration for the request processor
        region_name: AWS region to use for Bedrock API calls
        use_inference_profile: Whether to use inference profiles instead of direct model IDs
    """

    # List of models that support the Converse API
    # These models should use Converse API by default for better chat capabilities
    CONVERSE_SUPPORTED_MODELS = [
        # Anthropic Claude models - Best support for chat through Converse API
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
        
        # Meta Llama models - Chat-optimized through Converse API
        "meta.llama3-1-8b-instruct-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "meta.llama3-1-405b-instruct-v1:0",
        "meta.llama3-2-1b-instruct-v1:0",
        "meta.llama3-2-3b-instruct-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
        "meta.llama3-2-90b-instruct-v1:0",
        "meta.llama3-3-70b-instruct-v1:0",
        
        # Mistral AI models - Chat-optimized through Converse API
        "mistral.mistral-small-2402-v1:0",
        "mistral.mistral-large-2407-v1:0",
        
        # Amazon models - Basic chat support through Converse API
        "amazon.titan-text-express-v1",
        "amazon.titan-text-premier-v1",
        "amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
        "amazon.nova-pro-v1:0"
    ]

    # List of cross-region inference profiles
    INFERENCE_PROFILES = [
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-micro-v1:0",
        "us.amazon.nova-pro-v1:0",
        "us.anthropic.claude-3-haiku-20240307-v1:0",
        "us.anthropic.claude-3-opus-20240229-v1:0",
        "us.anthropic.claude-3-sonnet-20240229-v1:0",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "us.meta.llama3-1-8b-instruct-v1:0",
        "us.meta.llama3-1-70b-instruct-v1:0",
        "us.meta.llama3-1-405b-instruct-v1:0",
        "us.meta.llama3-2-1b-instruct-v1:0",
        "us.meta.llama3-2-3b-instruct-v1:0",
        "us.meta.llama3-2-11b-instruct-v1:0",
        "us.meta.llama3-2-90b-instruct-v1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
    ]

    def __init__(
        self, 
        config: OnlineRequestProcessorConfig,
        region_name: str = None,
        use_inference_profile: bool = False,
    ):
        """Initialize the BedrockOnlineRequestProcessor."""
        super().__init__(config)
        
        # Initialize AWS Bedrock client
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region_name)
        self.bedrock = boto3.client('bedrock', region_name=self.region_name)
        
        # Store model ID and determine provider
        self.model_id = config.model
        self.use_inference_profile = use_inference_profile
        
        # If using inference profiles, check if the model ID is already a profile or needs conversion
        if self.use_inference_profile:
            self.model_id = self._get_inference_profile_id(self.model_id)
            
        self.model_provider = self._determine_model_provider()
        
        # Provider-specific settings
        self.max_image_size_mb = self._get_max_image_size()
        
        logger.info(f"Initialized AWS Bedrock processor for model {self.model_id} " 
                   f"({self.model_provider}) in region {self.region_name}")
        if self.use_inference_profile:
            logger.info(f"Using inference profile: {self.model_id}")

    @property
    def _multimodal_prompt_supported(self) -> bool:
        """Whether this processor supports multimodal prompts.
        
        Returns:
            True since all Bedrock models support multimodal prompts
        """
        return True

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
                
        # T ry to find a matching inference profile
        base_model_name = model_id.split(":")[0] if ":" in model_id else model_id
        
        for profile in self.INFERENCE_PROFILES:
            # Check if the profile contains the base model name
            if base_model_name in profile:
                logger.info(f"Converting {model_id} to inference profile: {profile}")
                return profile
                
        # No matching profile found, use the model ID as is
        logger.warning(f"No matching inference profile found for {model_id}, using standard model ID")
        return model_id

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

    def _get_max_image_size(self) -> float:
        """Get the maximum image size in MB for the model provider.
        
        Returns:
            Maximum image size in MB
        """
        if self.model_provider == "anthropic":
            return 5.0  # Claude models support up to 5MB per image
        elif self.model_provider == "amazon":
            return 5.0  # Amazon Titan models
        elif self.model_provider == "meta":
            return 5.0  # Meta Llama models
        else:
            return 4.0  # Default conservative limit for other providers
            
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

    def validate_config(self):
        """Validate configuration parameters.
        
        Makes sure required configuration options are present and valid.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.model_id:
            raise ValueError("Model ID is required for AWS Bedrock")

    def file_upload_limit_check(self, base64_image: str) -> None:
        """Check if the image size is within the allowed limit.
        
        Args:
            base64_image: Base64 encoded image data
            
        Raises:
            ValueError: If image size exceeds the limit
        """
        # Calculate size in MB (4/3 ratio for base64)
        size_mb = (len(base64_image) * 3/4) / (1024 * 1024)
        if size_mb > self.max_image_size_mb:
            raise ValueError(
                f"Image size {size_mb:.2f}MB exceeds {self.model_provider} limit of {self.max_image_size_mb}MB"
            )

    def estimate_total_tokens(self, messages: list) -> _TokenUsage:
        """Estimate token usage for the given messages.
        
        Args:
            messages: List of messages to estimate token count for
            
        Returns:
            TokenUsage object with input and output token estimates
        """
        # This is a simplistic estimate - for accuracy, we'd need to use model-specific tokenizers
        total_text = ""
        
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                total_text += item["text"]
                    elif isinstance(content, str):
                        total_text += content
                elif isinstance(message, str):
                    total_text += message
        elif isinstance(messages, str):
            total_text = messages
            
        # Rough estimate: 1 token ≈ 4 characters for English text
        input_tokens = len(total_text) // 4
        
        # Estimate output tokens - this will depend on the model and prompt
        # A rough heuristic is that output is often similar in size to input for many cases
        output_tokens = input_tokens
        
        return _TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def estimate_output_tokens(self) -> int:
        """Estimate the number of output tokens.
        
        Returns:
            Estimated number of output tokens
        """
        # Use moving average if available, otherwise a reasonable default
        if self._output_tokens_window:
            return self._output_tokens_moving_average()
        return 500  # Default reasonable estimate

    def _format_anthropic_request(self, generic_request: GenericRequest) -> dict:
        """Format a request for Anthropic Claude models.
        
        Args:
            generic_request: The generic request to format
            
        Returns:
            Dict containing the formatted request
        """
        # Convert to Claude messages format
        messages = generic_request.messages
        
        # Prepare the request body
        request_body = {
            "messages": messages,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": generic_request.generation_params.get("max_tokens", 1000)
        }
        
        # Add optional parameters
        if "temperature" in generic_request.generation_params:
            request_body["temperature"] = generic_request.generation_params["temperature"]
        if "top_p" in generic_request.generation_params:
            request_body["top_p"] = generic_request.generation_params["top_p"]
        if "top_k" in generic_request.generation_params:
            request_body["top_k"] = generic_request.generation_params["top_k"]
        if "stop_sequences" in generic_request.generation_params:
            request_body["stop_sequences"] = generic_request.generation_params["stop_sequences"]
            
        return request_body

    def _format_amazon_request(self, generic_request: GenericRequest) -> dict:
        """Format a request for Amazon Titan models.
        
        Args:
            generic_request: The generic request to format
            
        Returns:
            Dict containing the formatted request
        """
        # Extract generation parameters
        text_gen_config = {}
        for param_name, bedrock_name in [
            ("max_tokens", "maxTokenCount"),
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("stop_sequences", "stopSequences")
        ]:
            if param_name in generic_request.generation_params:
                text_gen_config[bedrock_name] = generic_request.generation_params[param_name]
        
        # Get the text from the messages
        text = " ".join(msg.get("content", "") for msg in generic_request.messages)
        
        return {
            "inputText": text,
            "textGenerationConfig": text_gen_config
        }

    def _format_meta_request(self, generic_request: GenericRequest) -> dict:
        """Format a request for Meta Llama models.
        
        Args:
            generic_request: The generic request to format
            
        Returns:
            Dict containing the formatted request
        """
        # Convert to Llama messages format
        messages = generic_request.messages
        
        # Prepare the request body
        request_body = {
            "messages": messages
        }
        
        # Add optional parameters
        if "temperature" in generic_request.generation_params:
            request_body["temperature"] = generic_request.generation_params["temperature"]
        if "top_p" in generic_request.generation_params:
            request_body["top_p"] = generic_request.generation_params["top_p"]
        if "max_tokens" in generic_request.generation_params:
            request_body["max_gen_len"] = generic_request.generation_params["max_tokens"]
            
        return request_body

    def _format_cohere_request(self, generic_request: GenericRequest) -> dict:
        """Format a request for Cohere models.
        
        Args:
            generic_request: The generic request to format
            
        Returns:
            Dict containing the formatted request
        """
        # Extract text from messages
        text = " ".join(msg.get("content", "") for msg in generic_request.messages)
        
        # Command format
        request_body = {
            "prompt": text
        }
        
        # Add optional parameters
        if "temperature" in generic_request.generation_params:
            request_body["temperature"] = generic_request.generation_params["temperature"]
        if "top_p" in generic_request.generation_params:
            request_body["p"] = generic_request.generation_params["top_p"]
        if "top_k" in generic_request.generation_params:
            request_body["k"] = generic_request.generation_params["top_k"]
        if "max_tokens" in generic_request.generation_params:
            request_body["max_tokens"] = generic_request.generation_params["max_tokens"]
        if "stop_sequences" in generic_request.generation_params:
            request_body["stop_sequences"] = generic_request.generation_params["stop_sequences"]
            
        return request_body

    def _format_ai21_request(self, generic_request: GenericRequest) -> dict:
        """Format a request for AI21 models.
        
        Args:
            generic_request: The generic request to format
            
        Returns:
            Dict containing the formatted request
        """
        # Jamba format (chat-style)
        request_body = {
            "messages": generic_request.messages
        }
        
        # Add penalty parameters if specified
        for penalty in ["countPenalty", "presencePenalty", "frequencyPenalty"]:
            if penalty in generic_request.generation_params:
                request_body[penalty] = {
                    "scale": generic_request.generation_params[penalty]
                }
        
        # Add optional parameters
        if "temperature" in generic_request.generation_params:
            request_body["temperature"] = generic_request.generation_params["temperature"]
        if "top_p" in generic_request.generation_params:
            request_body["topP"] = generic_request.generation_params["top_p"]
        if "max_tokens" in generic_request.generation_params:
            request_body["maxTokens"] = generic_request.generation_params["max_tokens"]
        if "stop_sequences" in generic_request.generation_params:
            request_body["stopSequences"] = generic_request.generation_params["stop_sequences"]
            
        return request_body

    def _format_mistral_request(self, generic_request: GenericRequest) -> dict:
        """Format a request for Mistral AI models.
        
        Args:
            generic_request: The generic request to format
            
        Returns:
            Dict containing the formatted request
        """
        # Convert to Mistral messages format
        messages = generic_request.messages
        
        # Prepare the request body
        request_body = {
            "messages": messages
        }
        
        # Add optional parameters
        if "temperature" in generic_request.generation_params:
            request_body["temperature"] = generic_request.generation_params["temperature"]
        if "top_p" in generic_request.generation_params:
            request_body["top_p"] = generic_request.generation_params["top_p"]
        if "max_tokens" in generic_request.generation_params:
            request_body["max_tokens"] = generic_request.generation_params["max_tokens"]
            
        return request_body

    def _format_converse_request(self, generic_request: GenericRequest) -> dict:
        """Format a request for the AWS Bedrock Converse API.
        
        The Converse API is the preferred method for chat-based interactions as it
        provides better support for multi-turn conversations and consistent message
        formatting across different model providers.
        
        Args:
            generic_request: The generic request to format
            
        Returns:
            Dict containing the formatted request for the Converse API
        """
        # Handle system message if provided
        system_message = None
        messages = []
        
        # Extract messages from the request
        for message in generic_request.messages:
            role = message.get("role", "user").lower()
            content = message.get("content", "")
            
            # Extract system message if present
            if role == "system":
                # Extract text from content if it's a list of content items
                if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                    system_message = content[0]["text"]
                else:
                    system_message = content if isinstance(content, str) else str(content)
                continue
            
            # Format the message
            messages.append({
                "role": role,
                "content": content if isinstance(content, list) else [{"text": content}]
            })
            
        # Create the request body
        request_body = {
            "messages": messages
        }
        
        # Add system message if present
        if system_message:
            request_body["system"] = system_message
            
        # Add inference parameters
        config = {}
        
        # Set default values for inference parameters
        params = generic_request.generation_params
        
        # Default values based on common model settings
        config["temperature"] = params.get("temperature", 0.7)  # Standard creative temperature
        config["topP"] = params.get("top_p", 1.0)  # Default to no nucleus sampling restriction
        config["maxTokens"] = params.get("max_tokens", 2000)  # Reasonable default length
        config["stopSequences"] = params.get("stop_sequences", [])  # Empty list by default
            
        # Add config if not empty
        if config:
            request_body["inferenceConfig"] = config
            
        return request_body

    def _extract_converse_response(self, response_body: dict) -> t.Tuple[str, _TokenUsage]:
        """Extract text response and token usage from the Converse API response.
        
        Args:
            response_body: The response body from the Converse API
            
        Returns:
            Tuple of (text_response, token_usage)
        """
        # Default values
        text_response = ""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        # Extract the response content
        if "output" in response_body:
            output = response_body["output"]
            
            # Get message content
            if "message" in output:
                message = output["message"]
                
                # Extract content from message
                if "content" in message:
                    content_items = message["content"]
                    
                    # Combine all text content
                    text_parts = []
                    for item in content_items:
                        if "text" in item:
                            text_parts.append(item["text"])
                            
                    text_response = "\n".join(text_parts)
        
        # Get usage information - this is at the top level of the response, not in output
        if "usage" in response_body:
            usage = response_body["usage"]
            # Field names follow camelCase convention in Bedrock API
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            total_tokens = usage.get("totalTokens", 0)
        
        # Create token usage object
        token_usage = _TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        
        return text_response, token_usage

    def _supports_converse_api(self) -> bool:
        """Determine if the model supports the Converse API.
        
        Returns:
            True if the model supports the Converse API, False otherwise
        """
        # Check if using inference profile
        if self.model_id.startswith("us."):
            # Strip "us." prefix to check against the list
            base_model_id = self.model_id[3:]
            for model in self.CONVERSE_SUPPORTED_MODELS:
                if base_model_id == model or model in base_model_id:
                    return True
            return False
            
        # For regular model IDs
        model_base = self.model_id.split(":")[0] if ":" in self.model_id else self.model_id
        
        for supported_model in self.CONVERSE_SUPPORTED_MODELS:
            supported_base = supported_model.split(":")[0] if ":" in supported_model else supported_model
            if model_base == supported_base or model_base in supported_model:
                return True
                
        return False

    def create_api_specific_request_online(self, generic_request: GenericRequest) -> dict:
        """Create a model-specific request from a generic request for online inference.
        
        Args:
            generic_request: The generic request to convert
            
        Returns:
            Dict containing the model-specific request
            
        Raises:
            ValueError: If model provider is not supported
        """
        # Handle multimodal prompts
        generic_request = self._unpack_multimodal(generic_request)
        
        # Always try to use the Converse API first for supported models
        # as it provides better chat capabilities and consistent interface
        if self._supports_converse_api():
            try:
                return self._format_converse_request(generic_request)
            except Exception as e:
                logger.warning(f"Failed to format request for Converse API, falling back to provider-specific format: {str(e)}")
        
        # Fall back to provider-specific formatting if Converse API is not supported or formatting failed
        if self.model_provider == "anthropic":
            return self._format_anthropic_request(generic_request)
        elif self.model_provider == "amazon":
            return self._format_amazon_request(generic_request)
        elif self.model_provider == "meta":
            return self._format_meta_request(generic_request)
        elif self.model_provider == "cohere":
            return self._format_cohere_request(generic_request)
        elif self.model_provider == "ai21":
            return self._format_ai21_request(generic_request)
        elif self.model_provider == "mistral":
            return self._format_mistral_request(generic_request)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def _process_response(self, generic_response: GenericResponse) -> list:
        """Process a generic response into a format expected by the parser.
        
        Args:
            generic_response: The generic response to process
            
        Returns:
            List of processed responses in the expected format
        """
        # Ensure we have a valid response object
        if not generic_response:
            return []
        
        # Try to get text from the response object
        text = ""
        if hasattr(generic_response, 'text'):
            text = generic_response.text
        # If text is not directly available, try to extract it from raw_response
        elif hasattr(generic_response, 'raw_response'):
            raw_response = generic_response.raw_response
            # Check if this is a Converse API response
            if "output" in raw_response and "message" in raw_response["output"]:
                # Extract text from Converse API response
                message = raw_response["output"]["message"]
                if "content" in message:
                    content_items = message["content"]
                    text_parts = []
                    for item in content_items:
                        if "text" in item:
                            text_parts.append(item["text"])
                    text = "\n".join(text_parts)
            # Otherwise extract text based on model provider
            elif self.model_provider == "anthropic":
                text = raw_response.get("content", [{}])[0].get("text", "")
            elif self.model_provider == "amazon":
                text = raw_response.get("outputText", "")
            elif self.model_provider == "meta":
                text = raw_response.get("generation", "")
            elif self.model_provider == "cohere":
                text = raw_response.get("generations", [{}])[0].get("text", "")
            elif self.model_provider == "ai21":
                text = raw_response.get("completions", [{}])[0].get("data", {}).get("text", "")
            elif self.model_provider == "mistral":
                text = raw_response.get("outputs", [{}])[0].get("text", "")
            
        # Create a properly formatted response dictionary
        return [{"response": text}]
        
    def generate(self, prompt: str | list) -> dict:
        """Generate a response for a single prompt.

        Args:
            prompt: The prompt to send to the model (string or list of messages)

        Returns:
            A dictionary containing the model's response with 'response' as the key
        """
        # Set up default prompt formatter if not already set
        if not hasattr(self, 'prompt_formatter'):
            def default_prompt_func(row):
                if isinstance(row, str):
                    return row
                return row.get("prompt", "")

            self.prompt_formatter = PromptFormatter(
                model_name=self.model_id,
                prompt_func=default_prompt_func
            )

        # Create a generic request
        generic_request = GenericRequest(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt,
            generation_params=self.config.generation_params or {},
            original_row={"prompt": prompt},  # Store the original prompt
            original_row_idx=0  # Single request always has index 0
        )

        # Create API request
        request = APIRequest(
            task_id=0,
            generic_request=generic_request,
            api_specific_request=self.create_api_specific_request_online(generic_request),
            attempts_left=self.config.max_retries,
            prompt_formatter=self.prompt_formatter
        )

        # Make the request
        response = run_in_event_loop(
            self.call_single_request(
                request=request,
                session=aiohttp.ClientSession(),
                status_tracker=OnlineStatusTracker(
                    token_limit_strategy=self.token_limit_strategy,
                    max_requests_per_minute=self.max_requests_per_minute,
                    max_tokens_per_minute=self.max_tokens_per_minute,
                    compatible_provider=self.compatible_provider
                )
            )
        )

        # Process the response to extract the text
        processed_responses = self._process_response(response)
        
        # Return the processed response suitable for the LLM class
        if processed_responses:
            return processed_responses[0]
        else:
            # Fallback if no processed responses
            logger.warning("No processed responses available, returning empty response")
            if hasattr(response, 'raw_response'):
                logger.debug(f"Raw response: {response.raw_response}")
            return {"response": ""}

    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Make a single API request to AWS Bedrock.
        
        Args:
            request: The API request to make
            
        Returns:
            GenericResponse containing the model's response
            
        Raises:
            Exception: If the API call fails
        """
        try:
            # Get the request body
            request_body = request.api_specific_request

            # Determine which API to use
            if self._supports_converse_api():
                # Use Converse API
                # Prepare parameters for converse API
                params = {
                    "modelId": self.model_id,
                    "messages": request_body["messages"]
                }
                
                # Add system message if present, formatted as a list of dictionaries with text field
                if request_body.get("system"):
                    params["system"] = [{"text": request_body["system"]}]
                    
                # Add inference config if present
                if request_body.get("inferenceConfig"):
                    params["inferenceConfig"] = {
                        "temperature": request_body["inferenceConfig"].get("temperature"),
                        "topP": request_body["inferenceConfig"].get("topP"),
                        "maxTokens": request_body["inferenceConfig"].get("maxTokens"),
                        "stopSequences": request_body["inferenceConfig"].get("stopSequences")
                    }
                
                response = self.bedrock_runtime.converse(**params)
                response_body = response
                text_response, token_usage = self._extract_converse_response(response_body)
            else:
                # Use provider-specific API
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body).encode()
                )
                response_body = json.loads(response["body"].read())
                
                # Extract response based on provider
                if self.model_provider == "anthropic":
                    text_response = response_body.get("content", [{}])[0].get("text", "")
                    token_usage = _TokenUsage(
                        input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
                        output_tokens=response_body.get("usage", {}).get("output_tokens", 0)
                    )
                elif self.model_provider == "amazon":
                    text_response = response_body.get("outputText", "")
                    token_usage = _TokenUsage(
                        input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
                        output_tokens=response_body.get("usage", {}).get("output_tokens", 0)
                    )
                elif self.model_provider == "meta":
                    text_response = response_body.get("generation", "")
                    token_usage = _TokenUsage(
                        input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
                        output_tokens=response_body.get("usage", {}).get("output_tokens", 0)
                    )
                elif self.model_provider == "cohere":
                    text_response = response_body.get("generations", [{}])[0].get("text", "")
                    token_usage = _TokenUsage(
                        input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
                        output_tokens=response_body.get("usage", {}).get("output_tokens", 0)
                    )
                elif self.model_provider == "ai21":
                    text_response = response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
                    token_usage = _TokenUsage(
                        input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
                        output_tokens=response_body.get("usage", {}).get("output_tokens", 0)
                    )
                elif self.model_provider == "mistral":
                    text_response = response_body.get("outputs", [{}])[0].get("text", "")
                    token_usage = _TokenUsage(
                        input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
                        output_tokens=response_body.get("usage", {}).get("output_tokens", 0)
                    )
                else:
                    raise ValueError(f"Unsupported model provider: {self.model_provider}")
            
            # Get current time for timestamps
            current_time = datetime.datetime.now(datetime.UTC)
            
            # Create and return the response
            return GenericResponse(
                text=text_response,
                token_usage=token_usage,
                raw_response=response_body,
                generic_request=request.generic_request,
                created_at=current_time,
                finished_at=current_time,
                original_response=response_body
            )
            
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            raise
