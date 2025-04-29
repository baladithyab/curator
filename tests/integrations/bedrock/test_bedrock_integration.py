"""Integration tests for AWS Bedrock processors.

These tests require valid AWS credentials with access to Bedrock.
They will be skipped if the required environment variables are not set.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import boto3
import pytest

from bespokelabs.curator.llm.llm import LLM
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig, BatchRequestProcessorConfig
from bespokelabs.curator.request_processor.online.bedrock_online_request_processor import BedrockOnlineRequestProcessor
from bespokelabs.curator.request_processor.batch.bedrock_batch_request_processor import BedrockBatchRequestProcessor


# Skip all tests if AWS credentials are not available
skip_if_no_aws_credentials = pytest.mark.skipif(
    not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")),
    reason="AWS credentials not available"
)

# Skip batch tests if S3 bucket and role ARN are not available
skip_if_no_batch_config = pytest.mark.skipif(
    not (os.environ.get("BEDROCK_BATCH_S3_BUCKET") and os.environ.get("BEDROCK_BATCH_ROLE_ARN")),
    reason="Bedrock batch configuration not available"
)


@skip_if_no_aws_credentials
class TestBedrockOnlineIntegration(unittest.TestCase):
    """Integration tests for the BedrockOnlineRequestProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a Claude model for testing
        self.model_id = os.environ.get("BEDROCK_TEST_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
        self.region_name = os.environ.get("AWS_REGION", "us-east-1")
        
        # Create a config
        self.config = OnlineRequestProcessorConfig(
            model=self.model_id,
            generation_params={"temperature": 0.7, "max_tokens": 100}
        )
        
        # Create the processor
        self.processor = BedrockOnlineRequestProcessor(
            self.config,
            region_name=self.region_name
        )

    def test_generate_simple_prompt(self):
        """Test generating a response for a simple prompt."""
        # Skip if we're in CI environment
        if os.environ.get("CI"):
            self.skipTest("Skipping in CI environment")
            
        # Generate a response
        response = self.processor.generate("What is AWS Bedrock?")
        
        # Check that the response is not empty
        self.assertIn("response", response)
        self.assertTrue(response["response"])
        
        # Check that the response contains relevant information
        self.assertIn("Bedrock", response["response"])
        self.assertIn("AWS", response["response"])

    def test_generate_with_chat_messages(self):
        """Test generating a response with chat messages."""
        # Skip if we're in CI environment
        if os.environ.get("CI"):
            self.skipTest("Skipping in CI environment")
            
        # Generate a response
        response = self.processor.generate([
            {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
            {"role": "user", "content": "What is AWS Bedrock?"}
        ])
        
        # Check that the response is not empty
        self.assertIn("response", response)
        self.assertTrue(response["response"])
        
        # Check that the response contains relevant information
        self.assertIn("Bedrock", response["response"])
        self.assertIn("AWS", response["response"])

    def test_llm_integration(self):
        """Test integration with the LLM class."""
        # Skip if we're in CI environment
        if os.environ.get("CI"):
            self.skipTest("Skipping in CI environment")
            
        # Create an LLM instance
        llm = LLM(
            model=self.model_id,
            provider="bedrock",
            region_name=self.region_name
        )
        
        # Generate a response
        response = llm.generate("What is AWS Bedrock?")
        
        # Check that the response is not empty
        self.assertTrue(response)
        
        # Check that the response contains relevant information
        self.assertIn("Bedrock", response)
        self.assertIn("AWS", response)


@skip_if_no_aws_credentials
@skip_if_no_batch_config
class TestBedrockBatchIntegration(unittest.TestCase):
    """Integration tests for the BedrockBatchRequestProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a Claude model for testing
        self.model_id = os.environ.get("BEDROCK_TEST_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
        self.region_name = os.environ.get("AWS_REGION", "us-east-1")
        self.s3_bucket = os.environ.get("BEDROCK_BATCH_S3_BUCKET")
        self.s3_prefix = os.environ.get("BEDROCK_BATCH_S3_PREFIX", "curator-test")
        self.role_arn = os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        
        # Create a temporary directory for working files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a config
        self.config = BatchRequestProcessorConfig(
            model=self.model_id,
            generation_params={"temperature": 0.7, "max_tokens": 100},
            working_dir=self.temp_dir.name
        )
        
        # Create the processor
        self.processor = BedrockBatchRequestProcessor(
            self.config,
            region_name=self.region_name,
            s3_bucket=self.s3_bucket,
            s3_prefix=self.s3_prefix,
            role_arn=self.role_arn
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    @pytest.mark.asyncio
    async def test_batch_workflow(self):
        """Test the complete batch workflow."""
        # Skip if we're in CI environment
        if os.environ.get("CI"):
            pytest.skip("Skipping in CI environment")
            
        # Create a batch
        from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchStatus
        from bespokelabs.curator.types.generic_request import GenericRequest
        
        batch = GenericBatch(
            batch_id="test-batch",
            requests=[
                GenericRequest(
                    model=self.model_id,
                    prompt="What is AWS Bedrock?",
                    generation_params={"temperature": 0.7, "max_tokens": 100},
                    original_row={"prompt": "What is AWS Bedrock?"},
                    original_row_idx=0,
                    task_id=1
                ),
                GenericRequest(
                    model=self.model_id,
                    prompt="What is Amazon SageMaker?",
                    generation_params={"temperature": 0.7, "max_tokens": 100},
                    original_row={"prompt": "What is Amazon SageMaker?"},
                    original_row_idx=1,
                    task_id=2
                )
            ],
            status=GenericBatchStatus.PENDING
        )
        
        # Create a mock tracker
        self.processor.tracker = unittest.mock.AsyncMock()
        
        # Submit the batch
        await self.processor.submit_batch(batch)
        
        # Check that the batch was submitted
        self.assertEqual(batch.status, GenericBatchStatus.PROCESSING)
        self.assertTrue(batch.provider_batch_id)
        
        # Cancel the batch (to avoid waiting for completion)
        await self.processor.cancel_batch(batch)
        
        # Check that the batch was cancelled
        self.assertEqual(batch.status, GenericBatchStatus.FAILED)
        self.assertTrue(batch.metadata.get("cancelled", False))

    def test_llm_batch_integration(self):
        """Test integration with the LLM class for batch processing."""
        # Skip if we're in CI environment
        if os.environ.get("CI"):
            self.skipTest("Skipping in CI environment")
            
        # Create an LLM instance
        llm = LLM(
            model=self.model_id,
            provider="bedrock",
            region_name=self.region_name,
            batch_config={
                "s3_bucket": self.s3_bucket,
                "s3_prefix": self.s3_prefix,
                "role_arn": self.role_arn,
                "working_dir": self.temp_dir.name
            }
        )
        
        # Create a small dataset
        dataset = [
            {"prompt": "What is AWS Bedrock?"},
            {"prompt": "What is Amazon SageMaker?"}
        ]
        
        # Mock the batch processing to avoid actually running the job
        with patch.object(BedrockBatchRequestProcessor, 'submit_batch') as mock_submit_batch:
            # Generate batch responses
            llm.generate_batch(dataset)
            
            # Check that submit_batch was called
            mock_submit_batch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
