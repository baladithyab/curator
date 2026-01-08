"""Bedrock batch request processor for async batch inference.

Uses CreateModelInvocationJob API with S3 for input/output.
"""

import asyncio
import datetime
import json
import os
import uuid
from typing import Dict, List, Optional

from bespokelabs.curator.cost import cost_processor_factory
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.batch.base_batch_request_processor import (
    BaseBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.bedrock.bedrock_availability import (
    BATCH_INFERENCE_REGIONS,
    BedrockQuotas,
    get_available_batch_regions,
    get_batch_quotas,
    is_batch_available,
)
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import (
    GenericBatch,
    GenericBatchRequestCounts,
    GenericBatchStatus,
)
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


class BedrockBatchRequestProcessor(BaseBatchRequestProcessor):
    """Bedrock-specific implementation of the BatchRequestProcessor.

    Handles batch inference jobs using CreateModelInvocationJob with S3 storage.

    Note:
        - Requires boto3 and valid AWS credentials
        - Requires S3 bucket access for input/output data
        - Not all models/regions support batch inference
    """

    def __init__(self, config: BatchRequestProcessorConfig):
        """Initialize the BedrockBatchRequestProcessor."""
        super().__init__(config)

        self._compatible_provider = "bedrock"
        self._cost_processor = cost_processor_factory(config=config, backend=self._compatible_provider, batch=True)

        # AWS configuration
        self.region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self.profile = os.getenv("AWS_PROFILE")

        # S3 configuration for batch jobs
        self.s3_bucket = os.getenv("BEDROCK_BATCH_S3_BUCKET")
        self.s3_prefix = os.getenv("BEDROCK_BATCH_S3_PREFIX", "curator-batch-jobs")

        # IAM role for batch jobs
        self.batch_role_arn = os.getenv("BEDROCK_BATCH_ROLE_ARN")

        # Validate configuration
        self._validate_batch_config()

        # Initialize clients lazily
        self._bedrock_client = None
        self._s3_client = None

        # Load quotas dynamically with static fallback
        self._quotas: Optional[BedrockQuotas] = None

    def _validate_batch_config(self):
        """Validate batch processing configuration."""
        if not self.s3_bucket:
            raise ValueError(
                "BEDROCK_BATCH_S3_BUCKET environment variable is required for Bedrock batch processing. "
                "Set it to an S3 bucket name where batch input/output files will be stored."
            )

        if not self.batch_role_arn:
            raise ValueError(
                "BEDROCK_BATCH_ROLE_ARN environment variable is required for Bedrock batch processing. "
                "Set it to an IAM role ARN with permissions for Bedrock batch inference."
            )

        # Check if region supports batch inference
        if self.region not in BATCH_INFERENCE_REGIONS:
            available_regions = ", ".join(BATCH_INFERENCE_REGIONS[:5]) + "..."
            raise ValueError(
                f"Region {self.region} does not support Bedrock batch inference. "
                f"Supported regions include: {available_regions}"
            )

        # Check if model supports batch in this region
        model_id = self.config.model
        if not is_batch_available(model_id, self.region):
            available = get_available_batch_regions(model_id)
            if available:
                raise ValueError(
                    f"Model {model_id} does not support batch inference in region {self.region}. "
                    f"Available regions: {', '.join(available)}"
                )
            else:
                logger.warning(
                    f"Cannot verify batch support for model {model_id} in region {self.region}. "
                    "Proceeding anyway - the API will return an error if unsupported."
                )

    @property
    def bedrock_client(self):
        """Lazily initialize and return the Bedrock client."""
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
    def s3_client(self):
        """Lazily initialize and return the S3 client."""
        if self._s3_client is None:
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
            self._s3_client = session.client("s3", region_name=self.region)
        return self._s3_client

    @property
    def backend(self) -> str:
        """Backend property."""
        return "bedrock"

    @property
    def compatible_provider(self) -> str:
        """Compatible provider property."""
        return self._compatible_provider

    @property
    def quotas(self) -> BedrockQuotas:
        """Get batch quotas with dynamic retrieval and caching."""
        if self._quotas is None:
            self._quotas = get_batch_quotas()
        return self._quotas

    @property
    def max_requests_per_batch(self) -> int:
        """Maximum requests per batch for Bedrock.

        Dynamically retrieved from Service Quotas API with static fallback.
        """
        return self.quotas.max_records_per_batch_job

    @property
    def max_bytes_per_batch(self) -> int:
        """Maximum bytes per batch for Bedrock.

        Dynamically retrieved from Service Quotas API with static fallback.
        """
        return self.quotas.max_input_file_size_bytes

    @property
    def max_concurrent_batch_operations(self) -> int:
        """Maximum concurrent batch operations.

        Dynamically retrieved from Service Quotas API with static fallback.
        """
        return self.quotas.max_concurrent_batch_jobs

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> Dict:
        """Convert generic request to Bedrock batch format.

        Bedrock batch uses JSONL with recordId and modelInput fields.
        """
        # Build the model input based on provider
        model_id = generic_request.model
        provider = model_id.split(".")[0] if "." in model_id else ""

        if provider == "anthropic":
            model_input = self._build_anthropic_batch_input(generic_request)
        else:
            # Use Converse-style format for other models
            model_input = self._build_converse_batch_input(generic_request)

        return {
            "recordId": str(generic_request.original_row_idx),
            "modelInput": model_input,
        }

    def _build_anthropic_batch_input(self, generic_request: GenericRequest) -> Dict:
        """Build Anthropic-style batch input."""
        messages = []
        system_content = None

        for msg in generic_request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
                continue

            # Convert content format
            if isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })
            elif isinstance(content, list):
                formatted_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            formatted_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Handle image for batch
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                            if url.startswith("data:"):
                                # Base64 image
                                header, b64_data = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                formatted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": b64_data,
                                    },
                                })
                    elif isinstance(item, str):
                        formatted_content.append({"type": "text", "text": item})

                messages.append({"role": role, "content": formatted_content})

        model_input = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": generic_request.generation_params.get("max_tokens", 4096),
            "messages": messages,
        }

        if system_content:
            model_input["system"] = system_content

        if "temperature" in generic_request.generation_params:
            model_input["temperature"] = generic_request.generation_params["temperature"]

        if "top_p" in generic_request.generation_params:
            model_input["top_p"] = generic_request.generation_params["top_p"]

        return model_input

    def _build_converse_batch_input(self, generic_request: GenericRequest) -> Dict:
        """Build Converse-style batch input."""
        messages = []
        system_prompts = []

        for msg in generic_request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_prompts.append({"text": content})
                continue

            # Convert content
            if isinstance(content, str):
                converse_content = [{"text": content}]
            elif isinstance(content, list):
                converse_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            converse_content.append({"text": item.get("text", "")})
                    elif isinstance(item, str):
                        converse_content.append({"text": item})
            else:
                converse_content = [{"text": str(content)}]

            messages.append({
                "role": "user" if role == "user" else "assistant",
                "content": converse_content,
            })

        model_input = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": generic_request.generation_params.get("max_tokens", 4096),
            },
        }

        if system_prompts:
            model_input["system"] = system_prompts

        if "temperature" in generic_request.generation_params:
            model_input["inferenceConfig"]["temperature"] = generic_request.generation_params["temperature"]

        if "top_p" in generic_request.generation_params:
            model_input["inferenceConfig"]["topP"] = generic_request.generation_params["top_p"]

        return model_input

    async def submit_batch(
        self, requests: List[Dict], metadata: Optional[Dict] = None
    ) -> GenericBatch:
        """Submit a batch of requests to Bedrock.

        Args:
            requests: List of API-specific request dictionaries
            metadata: Optional metadata for the batch

        Returns:
            GenericBatch with job information
        """
        async with self.semaphore:
            # Generate unique job ID
            job_id = f"curator-{uuid.uuid4().hex[:8]}"

            # Upload requests to S3 as JSONL
            s3_input_uri = await self._upload_requests_to_s3(requests, job_id)

            # Create output location
            s3_output_uri = f"s3://{self.s3_bucket}/{self.s3_prefix}/{job_id}/output/"

            # Submit the batch job
            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.bedrock_client.create_model_invocation_job(
                        jobName=job_id,
                        modelId=self.config.model,
                        roleArn=self.batch_role_arn,
                        inputDataConfig={
                            "s3InputDataConfig": {
                                "s3Uri": s3_input_uri,
                            }
                        },
                        outputDataConfig={
                            "s3OutputDataConfig": {
                                "s3Uri": s3_output_uri,
                            }
                        },
                    ),
                )

                job_arn = response["jobArn"]
                logger.info(f"Submitted Bedrock batch job: {job_arn}")

                return GenericBatch(
                    id=job_arn,
                    created_at=datetime.datetime.now(),
                    status=GenericBatchStatus.SUBMITTED.value,
                    raw_status="Submitted",
                    request_counts=GenericBatchRequestCounts(
                        total=len(requests),
                        succeeded=0,
                        failed=0,
                    ),
                    request_file=metadata.get("request_file", "") if metadata else "",
                    api_specific_data={
                        "job_name": job_id,
                        "s3_input_uri": s3_input_uri,
                        "s3_output_uri": s3_output_uri,
                    },
                    attempts_left=self.config.max_retries,
                )

            except Exception as e:
                logger.error(f"Failed to submit Bedrock batch job: {e}")
                raise

    async def _upload_requests_to_s3(self, requests: List[Dict], job_id: str) -> str:
        """Upload requests to S3 as JSONL.

        Args:
            requests: List of request dictionaries
            job_id: Unique job identifier

        Returns:
            S3 URI of the uploaded file
        """
        # Create JSONL content
        jsonl_content = "\n".join(json.dumps(req) for req in requests)

        # Upload to S3
        s3_key = f"{self.s3_prefix}/{job_id}/input/requests.jsonl"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=jsonl_content.encode("utf-8"),
                ContentType="application/jsonl",
            ),
        )

        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        logger.debug(f"Uploaded batch input to {s3_uri}")

        return f"s3://{self.s3_bucket}/{self.s3_prefix}/{job_id}/input/"

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieve current status of a batch job.

        Args:
            batch: The batch to check

        Returns:
            Updated GenericBatch with current status
        """
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.bedrock_client.get_model_invocation_job(jobArn=batch.id),
            )

            # Map Bedrock status to generic status
            bedrock_status = response.get("status", "Unknown")
            status_mapping = {
                "Submitted": GenericBatchStatus.SUBMITTED.value,
                "InProgress": GenericBatchStatus.SUBMITTED.value,
                "Completed": GenericBatchStatus.FINISHED.value,
                "Failed": GenericBatchStatus.FAILED.value,
                "Stopping": GenericBatchStatus.SUBMITTED.value,
                "Stopped": GenericBatchStatus.FAILED.value,
                "PartiallyCompleted": GenericBatchStatus.FINISHED.value,
                "Expired": GenericBatchStatus.EXPIRED.value,
            }

            generic_status = status_mapping.get(bedrock_status, GenericBatchStatus.SUBMITTED.value)

            # Get request counts if available
            output_count = response.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("recordCount", 0)

            # For completed jobs, succeeded = output_count
            if generic_status == GenericBatchStatus.FINISHED.value:
                succeeded = output_count or batch.request_counts.total
                failed = batch.request_counts.total - succeeded
            else:
                succeeded = 0
                failed = 0

            return GenericBatch(
                id=batch.id,
                created_at=batch.created_at,
                status=generic_status,
                raw_status=bedrock_status,
                request_counts=GenericBatchRequestCounts(
                    total=batch.request_counts.total,
                    succeeded=succeeded,
                    failed=failed,
                ),
                request_file=batch.request_file,
                api_specific_data=batch.api_specific_data,
                attempts_left=batch.attempts_left,
            )

        except Exception as e:
            logger.error(f"Failed to retrieve batch status: {e}")
            return batch

    async def cancel_batch(self, batch: GenericBatch) -> GenericBatch:
        """Cancel a running batch job.

        Args:
            batch: The batch to cancel

        Returns:
            Updated GenericBatch after cancellation
        """
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: self.bedrock_client.stop_model_invocation_job(jobArn=batch.id),
            )
            logger.info(f"Cancelled Bedrock batch job: {batch.id}")

        except Exception as e:
            logger.warning(f"Failed to cancel batch job {batch.id}: {e}")

        return await self.retrieve_batch(batch)

    async def download_batch(self, batch: GenericBatch) -> Optional[str]:
        """Download results of a completed batch.

        Args:
            batch: The completed batch

        Returns:
            Raw response content as string, or None if failed
        """
        loop = asyncio.get_event_loop()

        try:
            # Get the output S3 location
            s3_output_uri = batch.api_specific_data.get("s3_output_uri", "")
            if not s3_output_uri:
                logger.error("No output S3 URI found for batch")
                return None

            # Parse S3 URI
            # Format: s3://bucket/prefix/
            parts = s3_output_uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""

            # List objects in the output location
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix),
            )

            # Find the output JSONL file
            contents = response.get("Contents", [])
            output_files = [obj for obj in contents if obj["Key"].endswith(".jsonl")]

            if not output_files:
                logger.error(f"No output files found at {s3_output_uri}")
                return None

            # Download all output files and concatenate
            all_results = []
            for obj in output_files:
                file_response = await loop.run_in_executor(
                    None,
                    lambda key=obj["Key"]: self.s3_client.get_object(Bucket=bucket, Key=key),
                )
                content = file_response["Body"].read().decode("utf-8")
                all_results.append(content)

            return "\n".join(all_results)

        except Exception as e:
            logger.error(f"Failed to download batch results: {e}")
            return None

    def parse_api_specific_response(
        self,
        raw_response: Dict,
        generic_request: GenericRequest,
        batch: GenericBatch,
    ) -> GenericResponse:
        """Parse Bedrock batch response into generic format.

        Args:
            raw_response: Raw response from Bedrock batch output
            generic_request: Original request
            batch: Batch context

        Returns:
            GenericResponse
        """
        # Bedrock batch output format:
        # {"recordId": "...", "modelOutput": {...}}

        model_output = raw_response.get("modelOutput", {})

        # Extract response text based on provider
        model_id = self.config.model
        provider = model_id.split(".")[0] if "." in model_id else ""

        if provider == "anthropic":
            content = model_output.get("content", [])
            response_text = ""
            for block in content:
                if block.get("type") == "text":
                    response_text = block.get("text", "")
                    break
        else:
            # Converse-style output
            output = model_output.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])
            response_text = ""
            for block in content:
                if "text" in block:
                    response_text = block["text"]
                    break

        # Handle return_completions_object flag
        if self.config.return_completions_object:
            response_message = model_output
        else:
            response_message = response_text

        # Get usage
        usage = model_output.get("usage", {})
        token_usage = _TokenUsage(
            input=usage.get("inputTokens", usage.get("input_tokens", 0)),
            output=usage.get("outputTokens", usage.get("output_tokens", 0)),
        )

        # Get stop reason
        finish_reason = model_output.get("stopReason", model_output.get("stop_reason", "stop"))

        # Calculate cost
        cost = self.completion_cost({"usage": {"inputTokens": token_usage.input, "outputTokens": token_usage.output}})

        return GenericResponse(
            response_message=response_message,
            response_errors=None,
            raw_request=None,  # Not available in batch
            raw_response=raw_response,
            generic_request=generic_request,
            created_at=batch.created_at,
            finished_at=datetime.datetime.now(),
            token_usage=token_usage,
            response_cost=cost,
            finish_reason=finish_reason,
        )
