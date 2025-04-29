"""Unit tests for the AWS Bedrock batch request processor."""

import asyncio
import datetime
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import aiofiles
import boto3
import pytest
from botocore.exceptions import ClientError

from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.request_processor.batch.bedrock_batch_request_processor import BedrockBatchRequestProcessor
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchStatus, GenericBatchRequestCounts
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage

# Create a mock BedrockBatchRequestProcessor with implementations for abstract methods
class MockBedrockBatchRequestProcessor(BedrockBatchRequestProcessor):
    """Mock implementation of BedrockBatchRequestProcessor for testing."""

    @property
    def max_bytes_per_batch(self) -> int:
        """Maximum size in bytes allowed for a single batch."""
        return 100 * 1024 * 1024  # 100 MB

    @property
    def max_concurrent_batch_operations(self) -> int:
        """Maximum number of concurrent batch operations."""
        return 10

    async def retrieve_batch(self, batch: GenericBatch) -> GenericBatch:
        """Retrieve current status of a submitted batch."""
        return batch

    async def download_batch(self, batch: GenericBatch) -> str | None:
        """Download results of a completed batch."""
        return '{"recordId": "1", "modelOutput": {"content": [{"type": "text", "text": "Test response"}]}}'

    def parse_api_specific_response(self, raw_response: dict, generic_request: GenericRequest, batch: GenericBatch) -> GenericResponse:
        """Parse API-specific response into standardized format."""
        return GenericResponse(
            task_id=1,
            response="Test response",
            success=True,
            finish_reason="stop",
            message="Success",
            processing_time=0,
            raw_data={},
            token_usage=_TokenUsage(input_tokens=10, output_tokens=20)
        )

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Convert generic request to API-specific format."""
        return {"prompt": generic_request.prompt}

    def parse_api_specific_batch_object(self, batch: object, request_file: str | None = None) -> GenericBatch:
        """Convert API-specific batch object to generic format."""
        return GenericBatch(
            batch_id="test-batch",
            requests=[],
            status=GenericBatchStatus.COMPLETE,
            provider_batch_id="test-job"
        )

    def parse_api_specific_request_counts(self, request_counts: object, request_file: str | None = None) -> GenericBatchRequestCounts:
        """Convert API-specific request counts to generic format."""
        return GenericBatchRequestCounts(
            total=1,
            succeeded=1,
            failed=0,
            pending=0
        )


class TestBedrockBatchRequestProcessor(unittest.TestCase):
    """Test suite for the BedrockBatchRequestProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock AWS credentials
        os.environ["AWS_ACCESS_KEY_ID"] = "test"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
        os.environ["AWS_REGION"] = "us-east-1"

        # Create a temporary directory for working files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a config with a Claude model
        self.config = BatchRequestProcessorConfig(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            generation_params={"temperature": 0.7, "max_tokens": 500}
        )

        # Create a patcher for boto3 client
        self.boto3_client_patcher = patch('boto3.client')
        self.mock_boto3_client = self.boto3_client_patcher.start()

        # Create mock clients
        self.mock_bedrock = MagicMock()
        self.mock_s3 = MagicMock()

        # Configure the mock to return our mock clients
        self.mock_boto3_client.side_effect = lambda service, **kwargs: {
            'bedrock': self.mock_bedrock,
            's3': self.mock_s3
        }[service]

        # Create the processor using our mock implementation
        self.processor = MockBedrockBatchRequestProcessor(
            self.config,
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
            role_arn="arn:aws:iam::123456789012:role/test-role"
        )

        # Set working_dir directly on the processor
        self.processor.working_dir = self.temp_dir.name

        # Create a mock tracker
        self.processor.tracker = AsyncMock()

    def tearDown(self):
        """Tear down test fixtures."""
        self.boto3_client_patcher.stop()
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the processor initializes correctly."""
        self.assertEqual(self.processor.model_id, "anthropic.claude-3-sonnet-20240229-v1:0")
        self.assertEqual(self.processor.model_provider, "anthropic")
        self.assertEqual(self.processor.region_name, "us-east-1")
        self.assertEqual(self.processor.s3_bucket, "test-bucket")
        self.assertEqual(self.processor.s3_prefix, "test-prefix")
        self.assertEqual(self.processor.role_arn, "arn:aws:iam::123456789012:role/test-role")
        self.assertFalse(self.processor.use_inference_profile)

    def test_inference_profile_conversion(self):
        """Test that model IDs are correctly converted to inference profiles."""
        # Create a processor with inference profile enabled
        processor = MockBedrockBatchRequestProcessor(
            self.config,
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
            role_arn="arn:aws:iam::123456789012:role/test-role",
            use_inference_profile=True
        )
        processor.working_dir = self.temp_dir.name

        # Check that the model ID was converted
        self.assertTrue(processor.model_id.startswith("us."))
        self.assertEqual(processor.model_id, "us.anthropic.claude-3-sonnet-20240229-v1:0")

    def test_validate_batch_support(self):
        """Test that the processor correctly validates batch support."""
        # Claude model should support batch
        self.processor._validate_batch_support()  # Should not raise an exception

        # Create a processor with a model that's not in the list
        config = BatchRequestProcessorConfig(
            model="unknown-model",
            generation_params={"temperature": 0.7, "max_tokens": 500}
        )

        # Mock the get_foundation_model response
        self.mock_bedrock.get_foundation_model.return_value = {
            "inferenceTypesSupported": ["ON_DEMAND"]
        }

        processor = MockBedrockBatchRequestProcessor(
            config,
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
            role_arn="arn:aws:iam::123456789012:role/test-role"
        )
        processor.working_dir = self.temp_dir.name

        # Should log a warning but not raise an exception
        processor._validate_batch_support()

    @pytest.mark.asyncio
    async def test_format_request_for_batch_anthropic(self):
        """Test that the processor correctly formats requests for Anthropic models."""
        # Create a generic request
        generic_request = GenericRequest(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            prompt="Tell me about AWS Bedrock.",
            generation_params={"temperature": 0.7, "max_tokens": 500},
            original_row={"prompt": "Tell me about AWS Bedrock."},
            original_row_idx=0,
            task_id=1
        )

        # Format the request
        request = await self.processor.format_request_for_batch(generic_request)

        # Check that the request is correctly formatted
        self.assertEqual(request["recordId"], "1")
        self.assertIn("modelInput", request)
        model_input = request["modelInput"]
        self.assertIn("messages", model_input)
        self.assertEqual(model_input["messages"][0]["role"], "user")
        self.assertEqual(model_input["messages"][0]["content"][0]["text"], "Tell me about AWS Bedrock.")
        self.assertEqual(model_input["temperature"], 0.7)
        self.assertEqual(model_input["max_tokens"], 500)
        self.assertEqual(model_input["anthropic_version"], "bedrock-2023-05-31")

    @pytest.mark.asyncio
    async def test_format_request_for_batch_amazon(self):
        """Test that the processor correctly formats requests for Amazon models."""
        # Create a processor with an Amazon model
        config = BatchRequestProcessorConfig(
            model="amazon.titan-text-express-v1",
            generation_params={"temperature": 0.7, "max_tokens": 500}
        )

        processor = MockBedrockBatchRequestProcessor(
            config,
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
            role_arn="arn:aws:iam::123456789012:role/test-role"
        )
        processor.working_dir = self.temp_dir.name

        # Create a generic request
        generic_request = GenericRequest(
            model="amazon.titan-text-express-v1",
            prompt="Tell me about AWS Bedrock.",
            generation_params={"temperature": 0.7, "max_tokens": 500},
            original_row={"prompt": "Tell me about AWS Bedrock."},
            original_row_idx=0,
            task_id=1
        )

        # Format the request
        request = await processor.format_request_for_batch(generic_request)

        # Check that the request is correctly formatted
        self.assertEqual(request["recordId"], "1")
        self.assertIn("modelInput", request)
        model_input = request["modelInput"]
        self.assertEqual(model_input["inputText"], "Tell me about AWS Bedrock.")
        self.assertIn("textGenerationConfig", model_input)
        self.assertEqual(model_input["textGenerationConfig"]["temperature"], 0.7)
        self.assertEqual(model_input["textGenerationConfig"]["maxTokenCount"], 500)

    @pytest.mark.asyncio
    async def test_submit_batch(self):
        """Test that the processor correctly submits a batch job."""
        # Create a batch
        batch = GenericBatch(
            batch_id="test-batch",
            requests=[
                GenericRequest(
                    model="anthropic.claude-3-sonnet-20240229-v1:0",
                    prompt="Tell me about AWS Bedrock.",
                    generation_params={"temperature": 0.7, "max_tokens": 500},
                    original_row={"prompt": "Tell me about AWS Bedrock."},
                    original_row_idx=0,
                    task_id=1
                )
            ],
            status=GenericBatchStatus.PENDING
        )

        # Mock the S3 upload_file method
        self.mock_s3.upload_file = MagicMock()

        # Mock the create_model_invocation_job method
        self.mock_bedrock.create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job"
        }

        # Create a patch for aiofiles.open
        with patch('aiofiles.open', new_callable=AsyncMock) as mock_aiofiles_open:
            # Configure the mock file
            mock_file = AsyncMock()
            mock_file.write = AsyncMock()
            mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

            # Submit the batch
            await self.processor.submit_batch(batch)

            # Check that the batch file was created
            mock_aiofiles_open.assert_called_once()

            # Check that the S3 upload was called
            self.mock_s3.upload_file.assert_called_once()

            # Check that the batch job was created
            self.mock_bedrock.create_model_invocation_job.assert_called_once()
            call_args = self.mock_bedrock.create_model_invocation_job.call_args[1]
            self.assertEqual(call_args["jobName"], "curator-batch-test-batch")
            self.assertEqual(call_args["modelId"], "anthropic.claude-3-sonnet-20240229-v1:0")
            self.assertEqual(call_args["roleArn"], "arn:aws:iam::123456789012:role/test-role")
            self.assertIn("inputDataConfig", call_args)
            self.assertIn("outputDataConfig", call_args)

            # Check that the batch was updated
            self.assertEqual(batch.status, GenericBatchStatus.PROCESSING)
            self.assertEqual(batch.provider_batch_id, "test-job")
            self.assertIn("job_id", batch.metadata)
            self.assertIn("job_arn", batch.metadata)
            self.assertIn("model_id", batch.metadata)
            self.assertIn("s3_input_uri", batch.metadata)
            self.assertIn("s3_output_uri", batch.metadata)

            # Check that the tracker was updated
            self.processor.tracker.append_batch.assert_called_once_with(batch)

    @pytest.mark.asyncio
    async def test_check_batch_status(self):
        """Test that the processor correctly checks batch status."""
        # Create a batch
        batch = GenericBatch(
            batch_id="test-batch",
            requests=[],
            status=GenericBatchStatus.PROCESSING,
            provider_batch_id="test-job",
            metadata={}
        )

        # Mock the get_model_invocation_job method
        self.mock_bedrock.get_model_invocation_job.return_value = {
            "status": "Completed"
        }

        # Check the batch status
        status = await self.processor.check_batch_status(batch)

        # Check that the status was updated
        self.assertEqual(status, GenericBatchStatus.COMPLETE)
        self.assertEqual(batch.metadata["bedrock_status"], "Completed")
        self.assertIn("last_checked", batch.metadata)

        # Check with a failed status
        self.mock_bedrock.get_model_invocation_job.return_value = {
            "status": "Failed",
            "failureMessage": "Test failure"
        }

        # Check the batch status
        status = await self.processor.check_batch_status(batch)

        # Check that the status was updated
        self.assertEqual(status, GenericBatchStatus.FAILED)
        self.assertEqual(batch.metadata["bedrock_status"], "Failed")
        self.assertEqual(batch.metadata["failure_reason"], "Test failure")

    @pytest.mark.asyncio
    async def test_fetch_batch_results(self):
        """Test that the processor correctly fetches batch results."""
        # Create a batch
        batch = GenericBatch(
            batch_id="test-batch",
            requests=[
                GenericRequest(
                    model="anthropic.claude-3-sonnet-20240229-v1:0",
                    prompt="Tell me about AWS Bedrock.",
                    generation_params={"temperature": 0.7, "max_tokens": 500},
                    original_row={"prompt": "Tell me about AWS Bedrock."},
                    original_row_idx=0,
                    task_id=1
                )
            ],
            status=GenericBatchStatus.COMPLETE,
            provider_batch_id="test-job",
            metadata={
                "s3_output_uri": "s3://test-bucket/test-prefix/output/test-batch"
            }
        )

        # Mock the S3 list_objects_v2 method
        self.mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "test-prefix/output/test-batch/results.jsonl.out"}
            ]
        }

        # Mock the S3 download_file method
        self.mock_s3.download_file = MagicMock()

        # Create a patch for aiofiles.open
        with patch('aiofiles.open', new_callable=AsyncMock) as mock_aiofiles_open:
            # Configure the mock file
            mock_file = AsyncMock()
            mock_file.read = AsyncMock(return_value=json.dumps({
                "recordId": "1",
                "modelOutput": {
                    "content": [{"type": "text", "text": "AWS Bedrock is a fully managed service."}],
                    "usage": {"input_tokens": 10, "output_tokens": 20}
                }
            }))
            mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

            # Fetch the batch results
            responses = await self.processor.fetch_batch_results(batch)

            # Check that the S3 methods were called
            self.mock_s3.list_objects_v2.assert_called_once()
            self.mock_s3.download_file.assert_called_once()

            # Check that the responses were correctly parsed
            self.assertEqual(len(responses), 1)
            self.assertEqual(responses[0].task_id, 1)
            self.assertEqual(responses[0].response, "AWS Bedrock is a fully managed service.")
            self.assertEqual(responses[0].token_usage.input_tokens, 10)
            self.assertEqual(responses[0].token_usage.output_tokens, 20)

    @pytest.mark.asyncio
    async def test_cancel_batch(self):
        """Test that the processor correctly cancels a batch job."""
        # Create a batch
        batch = GenericBatch(
            batch_id="test-batch",
            requests=[],
            status=GenericBatchStatus.PROCESSING,
            provider_batch_id="test-job",
            metadata={}
        )

        # Mock the stop_model_invocation_job method
        self.mock_bedrock.stop_model_invocation_job = MagicMock()

        # Create a patch for cleanup_batch
        with patch.object(self.processor, 'cleanup_batch', new_callable=AsyncMock) as mock_cleanup_batch:
            # Cancel the batch
            await self.processor.cancel_batch(batch)

            # Check that the stop_model_invocation_job method was called
            self.mock_bedrock.stop_model_invocation_job.assert_called_once_with(
                jobIdentifier="test-job"
            )

            # Check that the batch was updated
            self.assertEqual(batch.status, GenericBatchStatus.FAILED)
            self.assertTrue(batch.metadata["cancelled"])
            self.assertIn("cancelled_time", batch.metadata)

            # Check that cleanup_batch was called
            mock_cleanup_batch.assert_called_once_with(batch)

    @pytest.mark.asyncio
    async def test_cleanup_batch(self):
        """Test that the processor correctly cleans up batch resources."""
        # Create a batch
        batch = GenericBatch(
            batch_id="test-batch",
            requests=[],
            status=GenericBatchStatus.COMPLETE,
            provider_batch_id="test-job",
            metadata={}
        )

        # Add batch job info
        self.processor.batch_jobs["test-batch"] = {
            "batch_file_path": os.path.join(self.temp_dir.name, "test-batch_input.jsonl"),
            "input_s3_key": "test-prefix/input/test-batch_input.jsonl",
            "output_s3_uri": "s3://test-bucket/test-prefix/output/test-batch"
        }

        # Create the batch file
        with open(os.path.join(self.temp_dir.name, "test-batch_input.jsonl"), "w") as f:
            f.write('{"test": "data"}')

        # Enable deletion of successful batch files
        self.processor.config.delete_successful_batch_files = True

        # Mock the S3 list_objects_v2 method
        self.mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "test-prefix/output/test-batch/results.jsonl.out"}
            ]
        }

        # Clean up the batch
        await self.processor.cleanup_batch(batch)

        # Check that the batch file was deleted
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir.name, "test-batch_input.jsonl")))

        # Check that the S3 delete_object method was called
        self.mock_s3.delete_object.assert_called()

        # Check that the batch was removed from tracking
        self.assertNotIn("test-batch", self.processor.batch_jobs)


if __name__ == '__main__':
    unittest.main()
