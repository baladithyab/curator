"""AWS Bedrock request processor for batch inference.

This module provides a request processor that interfaces with AWS Bedrock
for batch inference requests.
"""

import asyncio
import datetime
import json
import os
import typing as t
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import boto3
from botocore.exceptions import ClientError

from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.batch.base_batch_request_processor import BaseBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchStatus
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


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
    """

    def __init__(
        self, 
        config: BatchRequestProcessorConfig,
        region_name: str = None,
        s3_bucket: str = None,
        s3_prefix: str = None,
        role_arn: str = None,
    ):
        """Initialize the BedrockBatchRequestProcessor."""
        super().__init__(config)
        
        # Initialize AWS clients
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.bedrock = boto3.client("bedrock", region_name=self.region_name)
        self.s3 = boto3.client("s3", region_name=self.region_name)
        
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
        self.model_provider = self._determine_model_provider()
        self.timeout_hours = 24  # Default timeout in hours
        self.batch_jobs = {}  # Track batch jobs: {batch_id: job_details}

    def _determine_model_provider(self) -> str:
        """Determine the model provider based on the model ID.
        
        Returns:
            The model provider as a string
        """
        model_id_lower = self.model_id.lower()
        
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
            if isinstance(generic_request.prompt, list):
                messages = []
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
                model_input = {
                    "inputText": generic_request.prompt if isinstance(generic_request.prompt, str) else str(generic_request.prompt)
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
                
                model_input = {
                    "inputText": generic_request.prompt if isinstance(generic_request.prompt, str) else str(generic_request.prompt),
                    "textGenerationConfig": text_gen_config
                }
                
        elif self.model_provider == "meta":
            # Meta Llama models
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
        return {
            "recordId": f"{generic_request.task_id}",
            "modelInput": model_input
        }

    async def submit_batch(self, batch: GenericBatch) -> None:
        """Submit a batch to AWS Bedrock.
        
        Args:
            batch: The batch to submit
            
        Raises:
            Exception: If batch submission fails
        """
        # Create batch input file in JSONL format
        batch_file_path = Path(self.working_dir) / f"{batch.batch_id}_input.jsonl"
        
        try:
            # Format requests for batch processing
            async with aiofiles.open(batch_file_path, "w") as f:
                for i, request in enumerate(batch.requests):
                    request_json = await self.format_request_for_batch(request)
                    await f.write(json.dumps(request_json) + "\n")
            
            logger.info(f"Created batch file with {len(batch.requests)} requests at {batch_file_path}")
            
            # Upload batch file to S3
            input_s3_key = f"{self.s3_prefix}/input/{batch.batch_id}_input.jsonl"
            await asyncio.to_thread(
                self.s3.upload_file, 
                str(batch_file_path), 
                self.s3_bucket, 
                input_s3_key
            )
            
            logger.info(f"Uploaded batch file to s3://{self.s3_bucket}/{input_s3_key}")
            
            # Set up S3 locations for input and output
            s3_input_uri = f"s3://{self.s3_bucket}/{input_s3_key}"
            s3_output_uri = f"s3://{self.s3_bucket}/{self.s3_prefix}/output/{batch.batch_id}"
            
            # Create batch job in AWS Bedrock
            job_response = await asyncio.to_thread(
                self.bedrock.create_model_invocation_job,
                jobName=f"curator-batch-{batch.batch_id}",
                modelId=self.model_id,
                roleArn=self.role_arn,
                inputDataConfig={
                    "s3InputDataConfig": {
                        "s3Uri": s3_input_uri
                    }
                },
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": s3_output_uri
                    }
                },
                timeoutDurationInHours=self.timeout_hours
            )
            
            # Store job details
            job_id = job_response.get("jobArn").split("/")[-1]
            
            logger.info(f"Submitted batch job {job_id} for batch {batch.batch_id}")
            
            # Update batch with job details
            batch.provider_batch_id = job_id
            batch.status = GenericBatchStatus.PROCESSING
            batch.metadata = {
                "job_id": job_id,
                "job_arn": job_response.get("jobArn"),
                "model_id": self.model_id,
                "s3_input_uri": s3_input_uri,
                "s3_output_uri": s3_output_uri,
                "submission_time": datetime.datetime.now().isoformat()
            }
            
            self.batch_jobs[batch.batch_id] = {
                "job_id": job_id,
                "input_s3_key": input_s3_key,
                "output_s3_uri": s3_output_uri,
                "batch_file_path": str(batch_file_path)
            }
            
            # Update batch status in tracker file
            await self.tracker.append_batch(batch)
            
        except Exception as e:
            logger.error(f"Failed to submit batch {batch.batch_id}: {str(e)}")
            batch.status = GenericBatchStatus.FAILED
            batch.metadata = {"error": str(e)}
            await self.tracker.append_batch(batch)
            raise Exception(f"Failed to submit batch: {str(e)}")

    async def check_batch_status(self, batch: GenericBatch) -> GenericBatchStatus:
        """Check the status of a batch job in AWS Bedrock.
        
        Args:
            batch: The batch to check
            
        Returns:
            Updated batch status
        """
        if not batch.provider_batch_id:
            logger.error(f"Batch {batch.batch_id} has no provider batch ID")
            return GenericBatchStatus.FAILED
        
        try:
            response = await asyncio.to_thread(
                self.bedrock.get_model_invocation_job,
                jobIdentifier=batch.provider_batch_id
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
            
            # Update batch metadata with latest status
            if "metadata" not in batch or not isinstance(batch.metadata, dict):
                batch.metadata = {}
            
            batch.metadata.update({
                "bedrock_status": bedrock_status,
                "last_checked": datetime.datetime.now().isoformat()
            })
            
            # Handle failure messages
            if bedrock_status in ["Failed", "Stopped", "Expired"]:
                if "failureMessage" in response:
                    batch.metadata["failure_reason"] = response.get("failureMessage")
                    logger.error(f"Batch {batch.batch_id} failed: {response.get('failureMessage')}")
            
            # Return mapped status
            return status_mapping.get(bedrock_status, GenericBatchStatus.PROCESSING)
            
        except Exception as e:
            logger.error(f"Error checking batch {batch.batch_id} status: {str(e)}")
            return batch.status  # Preserve current status on error

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
            
            # List objects in output location
            response = await asyncio.to_thread(
                self.s3.list_objects_v2,
                Bucket=bucket,
                Prefix=prefix
            )
            
            # Create a mapping from record ID to response
            responses_by_id = {}
            
            # Process all output files
            for obj in response.get("Contents", []):
                key = obj["Key"]
                
                # Skip any non-response files (e.g., manifests)
                if not key.endswith(".jsonl.out"):
                    continue
                
                # Download and process response file
                output_file_path = Path(self.working_dir) / f"{batch.batch_id}_output_{Path(key).name}"
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
            logger.error(f"Error fetching results for batch {batch.batch_id}: {str(e)}")
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
                    logger.warning(f"Failed to delete batch file {batch_file_path}: {str(e)}")
            
            # Clean up S3 files
            if getattr(self.config, "delete_successful_batch_files", False) and batch.status == GenericBatchStatus.COMPLETE:
                try:
                    # Delete input file from S3
                    input_s3_key = job_info.get("input_s3_key")
                    if input_s3_key:
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
                        prefix = parsed_uri.path.lstrip("/")
                        
                        # List and delete output objects
                        response = await asyncio.to_thread(
                            self.s3.list_objects_v2,
                            Bucket=self.s3_bucket,
                            Prefix=prefix
                        )
                        
                        for obj in response.get("Contents", []):
                            await asyncio.to_thread(
                                self.s3.delete_object,
                                Bucket=self.s3_bucket,
                                Key=obj["Key"]
                            )
                        
                        logger.info(f"Deleted S3 output files: {output_s3_uri}")
                except Exception as e:
                    logger.warning(f"Failed to delete S3 files for batch {batch.batch_id}: {str(e)}")
            
            # Remove batch from tracking
            self.batch_jobs.pop(batch.batch_id, None)

    async def cancel_batch(self, batch: GenericBatch) -> None:
        """Cancel a batch job in AWS Bedrock.
        
        Args:
            batch: The batch to cancel
            
        Raises:
            Exception: If cancellation fails
        """
        if not batch.provider_batch_id:
            logger.error(f"Batch {batch.batch_id} has no provider batch ID")
            return
        
        try:
            # Stop the batch job
            await asyncio.to_thread(
                self.bedrock.stop_model_invocation_job,
                jobIdentifier=batch.provider_batch_id
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
            
        except Exception as e:
            logger.error(f"Failed to cancel batch {batch.batch_id}: {str(e)}")
            raise Exception(f"Failed to cancel batch: {str(e)}") 