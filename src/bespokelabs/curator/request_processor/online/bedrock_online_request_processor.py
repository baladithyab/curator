"""AWS Bedrock request processor for online inference.

This module provides a request processor that interfaces with AWS Bedrock
for real-time inference requests.
"""

import asyncio
import datetime
import json
import os
import re
import typing as t
from io import BytesIO

import aiohttp
import boto3
from botocore.exceptions import ClientError

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import (
    APIRequest,
    BaseOnlineRequestProcessor,
)
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage
from bespokelabs.curator.utils.image_utils import get_file_size, resize_image


class BedrockOnlineRequestProcessor(BaseOnlineRequestProcessor):
    """Request processor for AWS Bedrock API.

    This class handles making real-time inference requests to the AWS Bedrock API,
    with support for different model providers hosted on Bedrock.

    Args:
        config: Configuration for the request processor
        region_name: AWS region to use for Bedrock API calls
        use_inference_profile: Whether to use inference profiles instead of direct model IDs
    """

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
        self.client = boto3.client('bedrock-runtime', region_name=self.region_name)
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
        
        # Set up default prompt formatters based on model provider
        self.prompt_formatters = {
            "anthropic": PromptFormatter(
                pre_prompt_format="{prompt}",
                post_response_format="{response}",
            ),
            "amazon": PromptFormatter(
                pre_prompt_format="{prompt}",
                post_response_format="{response}",
            ),
            "meta": PromptFormatter(
                pre_prompt_format="{prompt}",
                post_response_format="{response}",
            ),
            "cohere": PromptFormatter(
                pre_prompt_format="{prompt}",
                post_response_format="{response}",
            ),
            "ai21": PromptFormatter(
                pre_prompt_format="{prompt}",
                post_response_format="{response}",
            ),
            "mistral": PromptFormatter(
                pre_prompt_format="{prompt}",
                post_response_format="{response}",
            ),
        }

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

    def file_upload_limit_check(self, file_paths: t.List[str]) -> bool:
        """Check if files meet size limits for the model.
        
        Args:
            file_paths: List of file paths to check
            
        Returns:
            True if all files meet size limits, False otherwise
        """
        for file_path in file_paths:
            file_size_mb = get_file_size(file_path) / (1024 * 1024)
            if file_size_mb > self.max_image_size_mb:
                logger.warning(
                    f"File {file_path} exceeds {self.model_provider} size limit: "
                    f"{file_size_mb:.2f}MB > {self.max_image_size_mb}MB"
                )
                return False
        return True

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
        if isinstance(generic_request.prompt, list):
            messages = []
            
            # Convert to Claude messages format
            for message in generic_request.prompt:
                if isinstance(message, dict):
                    # Already in message format
                    messages.append(message)
                else:
                    # Single string - treat as user message
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": message}]
                    })
        else:
            # Single string prompt
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": generic_request.prompt}]
            }]
        
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
        if "embed" in self.model_id.lower():
            # Titan Embeddings format
            return {
                "inputText": generic_request.prompt if isinstance(generic_request.prompt, str) else str(generic_request.prompt)
            }
        else:
            # Titan Text format
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
            
            return {
                "inputText": generic_request.prompt if isinstance(generic_request.prompt, str) else str(generic_request.prompt),
                "textGenerationConfig": text_gen_config
            }

    def _format_meta_request(self, generic_request: GenericRequest) -> dict:
        """Format a request for Meta Llama models.
        
        Args:
            generic_request: The generic request to format
            
        Returns:
            Dict containing the formatted request
        """
        if isinstance(generic_request.prompt, list):
            # Convert to Llama messages format
            messages = []
            for message in generic_request.prompt:
                if isinstance(message, dict):
                    # Already in message format
                    messages.append(message)
                else:
                    # Single string - treat as user message
                    messages.append({
                        "role": "user",
                        "content": str(message)
                    })
        else:
            # Single string prompt
            messages = [{
                "role": "user",
                "content": str(generic_request.prompt)
            }]
        
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
        if "embed" in self.model_id.lower():
            # Cohere Embed format
            return {
                "texts": [generic_request.prompt] if isinstance(generic_request.prompt, str) else [str(generic_request.prompt)],
                "input_type": "search_document",
                "truncate": "END"
            }
        elif "command-r" in self.model_id.lower():
            # Command R format
            if isinstance(generic_request.prompt, list):
                # Extract chat history
                chat_history = []
                for message in generic_request.prompt[:-1]:  # All but the last message
                    if isinstance(message, dict):
                        role = "USER" if message.get("role", "").lower() == "user" else "CHATBOT"
                        content = message.get("content", "")
                        chat_history.append({"role": role, "message": content})
                    else:
                        # Alternate messages as user/chatbot
                        role = "USER" if len(chat_history) % 2 == 0 else "CHATBOT"
                        chat_history.append({"role": role, "message": str(message)})
                
                # Get the last message as the current query
                last_message = generic_request.prompt[-1]
                if isinstance(last_message, dict):
                    query = last_message.get("content", "")
                else:
                    query = str(last_message)
            else:
                # Single string prompt, no chat history
                query = generic_request.prompt
                chat_history = []
            
            # Prepare the request body
            request_body = {
                "message": query,
                "chat_history": chat_history
            }
        else:
            # Command format
            prompt = generic_request.prompt
            if isinstance(prompt, list):
                # Concatenate list elements
                prompt = "\n".join([str(p) for p in prompt])
            
            request_body = {
                "prompt": prompt
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
        if "jamba" in self.model_id.lower():
            # Jamba format (chat-style)
            if isinstance(generic_request.prompt, list):
                messages = []
                for message in generic_request.prompt:
                    if isinstance(message, dict):
                        # Already in message format
                        messages.append(message)
                    else:
                        # Single string - treat as user message
                        messages.append({
                            "role": "user",
                            "content": str(message)
                        })
            else:
                # Single string prompt
                messages = [{
                    "role": "user",
                    "content": str(generic_request.prompt)
                }]
            
            request_body = {
                "messages": messages
            }
        else:
            # Jurassic format
            prompt = generic_request.prompt
            if isinstance(prompt, list):
                # Concatenate list elements
                prompt = "\n".join([str(p) for p in prompt])
            
            request_body = {
                "prompt": prompt
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
        if isinstance(generic_request.prompt, list):
            messages = []
            for message in generic_request.prompt:
                if isinstance(message, dict):
                    # Already in message format
                    messages.append(message)
                else:
                    # Single string - treat as user message
                    messages.append({
                        "role": "user",
                        "content": str(message)
                    })
        else:
            # Single string prompt
            messages = [{
                "role": "user",
                "content": str(generic_request.prompt)
            }]
        
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
        
        # Format request based on model provider
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

    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Call a single API request.
        
        Args:
            request: The API request to make
            session: The aiohttp session to use
            status_tracker: Status tracker to update
            
        Returns:
            GenericResponse containing the model's response
            
        Raises:
            Exception: If the request fails
        """
        start_time = datetime.datetime.now()
        api_request = json.dumps(request.api_specific_request)
        
        try:
            # Call Bedrock API using boto3 (synchronously, but within an async context)
            response = await asyncio.to_thread(
                self.bedrock_runtime.invoke_model,
                modelId=self.model_id,
                body=api_request,
                contentType="application/json",
                accept="application/json",
            )
            
            # Process the response body
            response_body = json.loads(response.get("body").read().decode("utf-8"))
            
            # Extract response based on model provider
            text_response, token_usage = self._extract_response_by_provider(response_body, request)
            
            # Calculate processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Create generic response
            generic_response = GenericResponse(
                task_id=request.task_id,
                response=text_response,
                success=True,
                finish_reason="stop",
                message="Success",
                processing_time=processing_time,
                completion_time=processing_time,
                raw_data=response_body,
                token_usage=token_usage
            )
            
            # Update token usage in status tracker
            status_tracker.update_token_usage("processed", token_usage)
            
            # Add token count to the moving window
            self._add_output_token_moving_window(token_usage.output_tokens)
            
            return generic_response
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            
            # Map error codes to appropriate handling
            if error_code in ["ThrottlingException", "ServiceQuotaExceededException"]:
                # Rate limiting errors
                logger.warning(f"Rate limit reached: {error_message}")
                raise Exception(f"Rate limit reached: {error_message}")
            else:
                # Other client errors
                logger.error(f"Bedrock API error: {error_code} - {error_message}")
                raise Exception(f"Bedrock API error: {error_code} - {error_message}")
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error calling Bedrock: {str(e)}")
            raise Exception(f"Unexpected error: {str(e)}")

    def _extract_response_by_provider(self, response_body: dict, request: APIRequest) -> t.Tuple[str, _TokenUsage]:
        """Extract text response and token usage from the response body based on model provider.
        
        Args:
            response_body: The raw response body from the API
            request: Original request for context
            
        Returns:
            Tuple of (text_response, token_usage)
            
        Raises:
            ValueError: If response format is not recognized
        """
        # Default values
        text_response = ""
        input_tokens = 0
        output_tokens = 0
        
        # Extract based on model provider
        if self.model_provider == "anthropic":
            # Claude models
            if "content" in response_body:
                for content_item in response_body.get("content", []):
                    if content_item.get("type") == "text":
                        text_response = content_item.get("text", "")
            
            # Get token usage
            usage = response_body.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
        elif self.model_provider == "amazon":
            # Amazon Titan models
            if "embed" in self.model_id.lower():
                # Embedding response doesn't have text output
                text_response = str(response_body.get("embedding", []))
                input_tokens = response_body.get("inputTextTokenCount", 0)
                output_tokens = 0
            else:
                # Text generation model
                text_response = response_body.get("results", [{}])[0].get("outputText", "")
                input_tokens = response_body.get("inputTextTokenCount", 0)
                output_tokens = response_body.get("results", [{}])[0].get("tokenCount", 0)
                
        elif self.model_provider == "meta":
            # Meta Llama models
            text_response = response_body.get("generation", "")
            input_tokens = response_body.get("prompt_token_count", 0)
            output_tokens = response_body.get("generation_token_count", 0)
            
        elif self.model_provider == "cohere":
            # Cohere models
            if "embed" in self.model_id.lower():
                # Embedding response
                text_response = str(response_body.get("embeddings", [[]])[0])
                input_tokens = len(response_body.get("texts", [""])[0]) // 4  # Rough estimate
                output_tokens = 0
            elif "command-r" in self.model_id.lower():
                # Command R format
                text_response = response_body.get("text", "")
                token_count = response_body.get("token_count", {})
                input_tokens = token_count.get("prompt_tokens", 0)
                output_tokens = token_count.get("completion_tokens", 0)
            else:
                # Command format
                text_response = response_body.get("generations", [{}])[0].get("text", "")
                # Rough token estimate as Cohere doesn't always provide token counts
                input_tokens = len(str(request.api_specific_request.get("prompt", ""))) // 4
                output_tokens = len(text_response) // 4
                
        elif self.model_provider == "ai21":
            # AI21 models
            if "jamba" in self.model_id.lower():
                # Jamba format
                text_response = response_body.get("message", {}).get("content", "")
                input_tokens = response_body.get("prompt_tokens", 0)
                output_tokens = response_body.get("completion_tokens", 0)
            else:
                # Jurassic format
                text_response = response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
                # AI21 doesn't provide token counts directly for Jurassic
                input_tokens = len(str(request.api_specific_request.get("prompt", ""))) // 4
                output_tokens = len(text_response) // 4
                
        elif self.model_provider == "mistral":
            # Mistral AI models
            choices = response_body.get("choices", [{}])
            if choices:
                message = choices[0].get("message", {})
                text_response = message.get("content", "")
            
            # Get token usage
            usage = response_body.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
        
        # Create token usage object
        token_usage = _TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        
        return text_response, token_usage 