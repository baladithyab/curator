"""Bedrock online request processor for real-time inference.

Uses the Converse API as primary with InvokeModel fallback for unsupported models.
"""

import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional, TypeVar

import aiohttp
import tiktoken

from bespokelabs.curator.cost import cost_processor_factory
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.bedrock.bedrock_availability import (
    has_limited_converse_support,
    requires_invoke_model,
    supports_converse_api,
)
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.base_online_request_processor import (
    APIRequest,
    BaseOnlineRequestProcessor,
)
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage

T = TypeVar("T")

_BEDROCK_ALLOWED_IMAGE_SIZE_MB = 20  # MB

# Models that support multimodal input
_BEDROCK_MULTIMODAL_SUPPORTED_PREFIXES = {
    "anthropic.claude-3",
    "amazon.nova",
    "meta.llama3-2-11b",
    "meta.llama3-2-90b",
    "meta.llama4",
}


class BedrockOnlineRequestProcessor(BaseOnlineRequestProcessor):
    """Bedrock-specific implementation of the OnlineRequestProcessor.

    Handles API requests to AWS Bedrock with rate limiting, token counting,
    and error handling. Uses Converse API as primary with InvokeModel fallback.

    Note:
        - Requires boto3 and valid AWS credentials
        - Automatically detects whether to use Converse or InvokeModel
        - Supports cross-region inference profiles
    """

    def __init__(self, config: OnlineRequestProcessorConfig):
        """Initialize the BedrockOnlineRequestProcessor."""
        super().__init__(config)

        self._compatible_provider = "bedrock"
        self._cost_processor = cost_processor_factory(config=config, backend=self._compatible_provider)

        # AWS configuration
        self.region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self.profile = os.getenv("AWS_PROFILE")

        # Initialize boto3 client lazily
        self._bedrock_client = None
        self._bedrock_runtime_client = None

        # Determine API mode based on model
        self._use_converse = self._should_use_converse()

        self.token_encoding = self.get_token_encoding()

    def _should_use_converse(self) -> bool:
        """Determine whether to use Converse API for this model."""
        model_id = self.config.model

        if requires_invoke_model(model_id):
            logger.info(f"Model {model_id} requires InvokeModel API")
            return False

        if has_limited_converse_support(model_id):
            logger.warning(f"Model {model_id} has limited Converse support (no chat history)")
            return True

        if supports_converse_api(model_id):
            logger.info(f"Model {model_id} supports Converse API")
            return True

        # Default to trying Converse first
        logger.info(f"Model {model_id} - attempting Converse API (will fallback to InvokeModel if needed)")
        return True

    @property
    def bedrock_runtime_client(self):
        """Lazily initialize and return the Bedrock Runtime client."""
        if self._bedrock_runtime_client is None:
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
            self._bedrock_runtime_client = session.client(
                "bedrock-runtime",
                region_name=self.region,
            )
        return self._bedrock_runtime_client

    @property
    def backend(self) -> str:
        """Backend property."""
        return "bedrock"

    @property
    def compatible_provider(self) -> str:
        """Compatible provider property."""
        return self._compatible_provider

    def file_upload_limit_check(self, base64_image: str) -> None:
        """Check if the image size is within the allowed limit."""
        from bespokelabs.curator.file_utilities import get_base64_size

        mb = get_base64_size(base64_image)
        if mb > _BEDROCK_ALLOWED_IMAGE_SIZE_MB:
            raise ValueError(
                f"Image size is {mb} MB, which is greater than the allowed size of "
                f"{_BEDROCK_ALLOWED_IMAGE_SIZE_MB} MB."
            )

    @property
    def _multimodal_prompt_supported(self) -> bool:
        """Check if the model supports multimodal prompts."""
        return any(
            self.config.model.startswith(prefix)
            for prefix in _BEDROCK_MULTIMODAL_SUPPORTED_PREFIXES
        )

    def get_token_encoding(self) -> "tiktoken.Encoding":
        """Get the token encoding for Bedrock models."""
        return tiktoken.get_encoding("cl100k_base")

    def estimate_output_tokens(self) -> int:
        """Estimate number of tokens in the response."""
        return int(self._output_tokens_moving_average()) or self._get_max_tokens() // 4

    def _get_max_tokens(self) -> int:
        """Get max tokens from config or default."""
        if self.config.generation_params.get("max_tokens"):
            return self.config.generation_params["max_tokens"]
        return 4096

    def estimate_total_tokens(self, messages: List[Dict]) -> _TokenUsage:
        """Estimate total tokens for a request.

        Args:
            messages: List of message dictionaries

        Returns:
            _TokenUsage with estimated input and output tokens
        """
        num_tokens = 0

        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                num_tokens += len(self.token_encoding.encode(content, disallowed_special=()))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            num_tokens += len(
                                self.token_encoding.encode(item.get("text", ""), disallowed_special=())
                            )
                        elif item.get("type") in ("image", "image_url"):
                            num_tokens += 1024  # Approximate for images

        num_tokens += 50  # Overhead for message formatting
        output_tokens = self.estimate_output_tokens()

        return _TokenUsage(input=num_tokens, output=output_tokens)

    def _format_multimodal_for_converse(self, data, mime_type: str = "image/png") -> Dict:
        """Format multimodal content for Converse API."""
        mime_type = mime_type or "image/png"

        if data.url and not data.is_local:
            # Converse API uses source.s3Location for S3 or bytes for inline
            return {
                "image": {
                    "format": mime_type.split("/")[-1],  # e.g., "png" from "image/png"
                    "source": {"bytes": None},  # Will need to fetch URL content
                }
            }
        else:
            import base64
            base64_content = data.serialize()
            self.file_upload_limit_check(base64_content)

            return {
                "image": {
                    "format": mime_type.split("/")[-1],
                    "source": {"bytes": base64.b64decode(base64_content)},
                }
            }

    def _convert_messages_to_converse_format(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict], Optional[List[Dict]]]:
        """Convert generic messages to Converse API format.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Tuple of (messages, system_prompts)
        """
        converse_messages = []
        system_prompts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # System messages are handled separately in Converse API
                if isinstance(content, str):
                    system_prompts.append({"text": content})
                continue

            # Map roles
            converse_role = "user" if role == "user" else "assistant"

            # Convert content
            if isinstance(content, str):
                converse_content = [{"text": content}]
            elif isinstance(content, list):
                converse_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            converse_content.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Handle image URL format
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else ""

                            if url.startswith("data:"):
                                # Base64 encoded image
                                import base64
                                # Parse data URL: data:image/png;base64,<data>
                                header, b64_data = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                image_format = mime_type.split("/")[-1]

                                converse_content.append({
                                    "image": {
                                        "format": image_format,
                                        "source": {"bytes": base64.b64decode(b64_data)},
                                    }
                                })
                        elif item.get("type") == "image":
                            # Already in Bedrock format
                            converse_content.append(item)
                    elif isinstance(item, str):
                        converse_content.append({"text": item})
            else:
                converse_content = [{"text": str(content)}]

            converse_messages.append({
                "role": converse_role,
                "content": converse_content,
            })

        return converse_messages, system_prompts if system_prompts else None

    def create_api_specific_request_online(self, generic_request: GenericRequest) -> Dict:
        """Create a Bedrock-specific request from a generic request.

        Args:
            generic_request: The generic request to convert

        Returns:
            API-specific request dictionary for Bedrock
        """
        if self._use_converse:
            return self._create_converse_request(generic_request)
        else:
            return self._create_invoke_model_request(generic_request)

    def _create_converse_request(self, generic_request: GenericRequest) -> Dict:
        """Create a Converse API request."""
        messages, system = self._convert_messages_to_converse_format(generic_request.messages)

        request = {
            "modelId": generic_request.model,
            "messages": messages,
        }

        if system:
            request["system"] = system

        # Build inference config
        inference_config = {}

        max_tokens = generic_request.generation_params.get("max_tokens", 4096)
        inference_config["maxTokens"] = max_tokens

        if "temperature" in generic_request.generation_params:
            inference_config["temperature"] = generic_request.generation_params["temperature"]

        if "top_p" in generic_request.generation_params:
            inference_config["topP"] = generic_request.generation_params["top_p"]

        if "stop" in generic_request.generation_params:
            stop_sequences = generic_request.generation_params["stop"]
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            inference_config["stopSequences"] = stop_sequences

        if inference_config:
            request["inferenceConfig"] = inference_config

        # Handle tool use if present
        if "tools" in generic_request.generation_params:
            request["toolConfig"] = {
                "tools": generic_request.generation_params["tools"]
            }

        return request

    def _create_invoke_model_request(self, generic_request: GenericRequest) -> Dict:
        """Create an InvokeModel API request.

        This handles model-specific body formats.
        """
        model_id = generic_request.model
        provider = model_id.split(".")[0] if "." in model_id else ""

        # Build the model-specific body
        if provider == "anthropic":
            body = self._build_anthropic_body(generic_request)
        elif provider == "amazon":
            body = self._build_amazon_body(generic_request)
        elif provider == "meta":
            body = self._build_meta_body(generic_request)
        elif provider == "mistral":
            body = self._build_mistral_body(generic_request)
        elif provider == "cohere":
            body = self._build_cohere_body(generic_request)
        else:
            # Generic format
            body = self._build_generic_body(generic_request)

        return {
            "modelId": model_id,
            "body": body,
            "contentType": "application/json",
            "accept": "application/json",
        }

    def _build_anthropic_body(self, generic_request: GenericRequest) -> Dict:
        """Build request body for Anthropic Claude models."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": generic_request.generation_params.get("max_tokens", 4096),
            "messages": generic_request.messages,
        }

        # Extract system message if present
        system_messages = [m for m in generic_request.messages if m.get("role") == "system"]
        if system_messages:
            body["system"] = system_messages[0].get("content", "")
            body["messages"] = [m for m in generic_request.messages if m.get("role") != "system"]

        if "temperature" in generic_request.generation_params:
            body["temperature"] = generic_request.generation_params["temperature"]

        if "top_p" in generic_request.generation_params:
            body["top_p"] = generic_request.generation_params["top_p"]

        return body

    def _build_amazon_body(self, generic_request: GenericRequest) -> Dict:
        """Build request body for Amazon Titan/Nova models."""
        # Extract the prompt from messages
        prompt = self._messages_to_prompt(generic_request.messages)

        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": generic_request.generation_params.get("max_tokens", 4096),
            },
        }

        if "temperature" in generic_request.generation_params:
            body["textGenerationConfig"]["temperature"] = generic_request.generation_params["temperature"]

        if "top_p" in generic_request.generation_params:
            body["textGenerationConfig"]["topP"] = generic_request.generation_params["top_p"]

        return body

    def _build_meta_body(self, generic_request: GenericRequest) -> Dict:
        """Build request body for Meta Llama models."""
        prompt = self._messages_to_prompt(generic_request.messages)

        body = {
            "prompt": prompt,
            "max_gen_len": generic_request.generation_params.get("max_tokens", 4096),
        }

        if "temperature" in generic_request.generation_params:
            body["temperature"] = generic_request.generation_params["temperature"]

        if "top_p" in generic_request.generation_params:
            body["top_p"] = generic_request.generation_params["top_p"]

        return body

    def _build_mistral_body(self, generic_request: GenericRequest) -> Dict:
        """Build request body for Mistral models."""
        prompt = self._messages_to_prompt(generic_request.messages)

        body = {
            "prompt": prompt,
            "max_tokens": generic_request.generation_params.get("max_tokens", 4096),
        }

        if "temperature" in generic_request.generation_params:
            body["temperature"] = generic_request.generation_params["temperature"]

        if "top_p" in generic_request.generation_params:
            body["top_p"] = generic_request.generation_params["top_p"]

        return body

    def _build_cohere_body(self, generic_request: GenericRequest) -> Dict:
        """Build request body for Cohere models."""
        prompt = self._messages_to_prompt(generic_request.messages)

        body = {
            "prompt": prompt,
            "max_tokens": generic_request.generation_params.get("max_tokens", 4096),
        }

        if "temperature" in generic_request.generation_params:
            body["temperature"] = generic_request.generation_params["temperature"]

        if "top_p" in generic_request.generation_params:
            body["p"] = generic_request.generation_params["top_p"]

        return body

    def _build_generic_body(self, generic_request: GenericRequest) -> Dict:
        """Build a generic request body."""
        prompt = self._messages_to_prompt(generic_request.messages)

        return {
            "prompt": prompt,
            "max_tokens": generic_request.generation_params.get("max_tokens", 4096),
        }

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to a single prompt string."""
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if isinstance(content, list):
                # Extract text from content list
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts)

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        parts.append("Assistant:")
        return "\n\n".join(parts)

    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Make a single Bedrock API request.

        Args:
            request: The request to process
            session: Async HTTP session (not used - using boto3)
            status_tracker: Tracks request status

        Returns:
            GenericResponse from Bedrock
        """
        try:
            if self._use_converse:
                response = await self._call_converse(request, status_tracker)
            else:
                response = await self._call_invoke_model(request, status_tracker)

            return response

        except Exception as e:
            error_str = str(e).lower()

            if "rate" in error_str or "throttl" in error_str:
                status_tracker.time_of_last_rate_limit_error = time.time()
                status_tracker.num_rate_limit_errors += 1

            raise

    async def _call_converse(
        self,
        request: APIRequest,
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Call the Converse API."""
        import asyncio

        api_request = request.api_specific_request.copy()
        model_id = api_request.pop("modelId")

        # Run synchronous boto3 call in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock_runtime_client.converse(
                modelId=model_id,
                **api_request,
            ),
        )

        # Parse response
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        # Extract text from content
        response_text = ""
        for block in content:
            if "text" in block:
                response_text = block["text"]
                break

        # Handle return_completions_object flag
        if self.config.return_completions_object:
            response_message = dict(response)
        else:
            response_message = response_text

        # Get usage
        usage = response.get("usage", {})
        token_usage = _TokenUsage(
            input=usage.get("inputTokens", 0),
            output=usage.get("outputTokens", 0),
        )

        # Get stop reason
        finish_reason = response.get("stopReason", "end_turn")
        # Map Bedrock stop reasons to standard format
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "content_filtered": "content_filter",
        }
        finish_reason = finish_reason_map.get(finish_reason, finish_reason)

        # Calculate cost
        cost = self.completion_cost(response)

        return GenericResponse(
            response_message=response_message,
            response_errors=None,
            raw_request=request.api_specific_request,
            raw_response=response,
            generic_request=request.generic_request,
            created_at=request.created_at,
            finished_at=datetime.datetime.now(),
            token_usage=token_usage,
            response_cost=cost,
            finish_reason=finish_reason,
        )

    async def _call_invoke_model(
        self,
        request: APIRequest,
        status_tracker: OnlineStatusTracker,
    ) -> GenericResponse:
        """Call the InvokeModel API."""
        import asyncio

        api_request = request.api_specific_request
        model_id = api_request["modelId"]
        body = api_request["body"]

        # Run synchronous boto3 call in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock_runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType=api_request.get("contentType", "application/json"),
                accept=api_request.get("accept", "application/json"),
            ),
        )

        # Parse response body
        response_body = json.loads(response["body"].read())

        # Extract response text based on provider
        provider = model_id.split(".")[0] if "." in model_id else ""
        response_text = self._extract_response_text(response_body, provider)

        # Handle return_completions_object flag
        if self.config.return_completions_object:
            response_message = response_body
        else:
            response_message = response_text

        # Estimate token usage (InvokeModel doesn't always return usage)
        input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
        output_tokens = response_body.get("usage", {}).get("output_tokens", 0)

        if input_tokens == 0:
            # Estimate based on request
            estimated = self.estimate_total_tokens(request.generic_request.messages)
            input_tokens = estimated.input

        if output_tokens == 0 and response_text:
            output_tokens = len(self.token_encoding.encode(response_text, disallowed_special=()))

        token_usage = _TokenUsage(input=input_tokens, output=output_tokens)

        # Get stop reason
        finish_reason = response_body.get("stop_reason", "stop")

        # Calculate cost
        cost = self.completion_cost({"usage": {"inputTokens": input_tokens, "outputTokens": output_tokens}})

        return GenericResponse(
            response_message=response_message,
            response_errors=None,
            raw_request=request.api_specific_request,
            raw_response=response_body,
            generic_request=request.generic_request,
            created_at=request.created_at,
            finished_at=datetime.datetime.now(),
            token_usage=token_usage,
            response_cost=cost,
            finish_reason=finish_reason,
        )

    def _extract_response_text(self, response_body: Dict, provider: str) -> str:
        """Extract response text from model-specific response format."""
        if provider == "anthropic":
            content = response_body.get("content", [])
            for block in content:
                if block.get("type") == "text":
                    return block.get("text", "")
            return ""

        elif provider == "amazon":
            results = response_body.get("results", [])
            if results:
                return results[0].get("outputText", "")
            return response_body.get("outputText", "")

        elif provider == "meta":
            return response_body.get("generation", "")

        elif provider == "mistral":
            outputs = response_body.get("outputs", [])
            if outputs:
                return outputs[0].get("text", "")
            return ""

        elif provider == "cohere":
            generations = response_body.get("generations", [])
            if generations:
                return generations[0].get("text", "")
            return response_body.get("text", "")

        else:
            # Try common patterns
            return (
                response_body.get("generation", "")
                or response_body.get("text", "")
                or response_body.get("completion", "")
                or str(response_body)
            )
