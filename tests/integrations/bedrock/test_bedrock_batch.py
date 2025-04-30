"""Integration tests for AWS Bedrock batch processing.

These tests verify that the Bedrock batch processor works correctly with
various models and configurations. They require valid AWS credentials
with access to Bedrock and appropriate S3 and IAM resources.
"""

import os
import json
import time
import tempfile
import unittest
from typing import Dict, List, Any, Optional

import boto3
import pytest

from bespokelabs import curator
from bespokelabs.curator.llm.llm import LLM
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.request_processor.batch.bedrock_batch_request_processor import BedrockBatchRequestProcessor
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchStatus

from tests.integrations.bedrock.utils import (
    check_aws_credentials,
    check_bedrock_access,
    setup_test_resources,
    cleanup_test_resources,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_LLAMA_MODEL
)


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
@skip_if_no_batch_config
class TestBedrockBatchIntegration(unittest.TestCase):
    """Integration tests for the BedrockBatchRequestProcessor."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        # Use environment variables if available, otherwise create test resources
        cls.s3_bucket = os.environ.get("BEDROCK_BATCH_S3_BUCKET")
        cls.role_arn = os.environ.get("BEDROCK_BATCH_ROLE_ARN")
        cls.created_resources = {}
        
        if not cls.s3_bucket or not cls.role_arn:
            # Create test resources
            cls.created_resources = setup_test_resources()
            cls.s3_bucket = cls.created_resources.get("s3_bucket")
            cls.role_arn = cls.created_resources.get("role_arn")
            
            if not cls.s3_bucket or not cls.role_arn:
                raise unittest.SkipTest("Failed to create required AWS resources")

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests have run."""
        # Only clean up resources we created
        if cls.created_resources:
            cleanup_test_resources(cls.created_resources)

    def setUp(self):
        """Set up test fixtures before each test."""
        # Use a Claude model for testing
        self.model_id = os.environ.get("BEDROCK_TEST_MODEL", DEFAULT_CLAUDE_MODEL)
        self.region_name = os.environ.get("AWS_REGION", "us-east-1")
        self.s3_prefix = os.environ.get("BEDROCK_BATCH_S3_PREFIX", "curator-test")
        
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
            config=self.config,
            region_name=self.region_name,
            s3_bucket=self.s3_bucket,
            s3_prefix=self.s3_prefix,
            role_arn=self.role_arn
        )

    def tearDown(self):
        """Clean up test fixtures after each test."""
        self.temp_dir.cleanup()

    def test_create_api_specific_request_batch(self):
        """Test creating API-specific request for batch processing."""
        # Create a generic request
        generic_request = GenericRequest(
            model=self.model_id,
            prompt="What is AWS Bedrock?",
            generation_params={"temperature": 0.7, "max_tokens": 100},
            original_row={"prompt": "What is AWS Bedrock?"},
            original_row_idx=0,
            task_id=1
        )
        
        # Convert to API-specific format
        api_request = self.processor.create_api_specific_request_batch(generic_request)
        
        # Check that the request has the expected format
        self.assertIn("recordId", api_request)
        self.assertIn("modelInput", api_request)
        self.assertEqual(api_request["recordId"], "1")  # Should use task_id
        
        # Check model-specific formatting
        model_input = api_request["modelInput"]
        if "anthropic" in self.model_id:
            self.assertIn("anthropic_version", model_input)
            self.assertIn("messages", model_input)
        elif "meta" in self.model_id:
            self.assertIn("prompt", model_input)
        elif "amazon" in self.model_id:
            self.assertIn("inputText", model_input)

    @pytest.mark.asyncio
    async def test_submit_batch(self):
        """Test submitting a batch job to Bedrock."""
        # Create a list of generic requests
        requests = [
            GenericRequest(
                model=self.model_id,
                prompt=f"What is AWS Bedrock? (Request {i})",
                generation_params={"temperature": 0.7, "max_tokens": 100},
                original_row={"prompt": f"What is AWS Bedrock? (Request {i})"},
                original_row_idx=i,
                task_id=i
            )
            for i in range(3)
        ]
        
        # Convert to API-specific format
        api_requests = [
            self.processor.create_api_specific_request_batch(req)
            for req in requests
        ]
        
        # Submit the batch
        batch = await self.processor.submit_batch(
            requests=api_requests,
            metadata={"test_id": "test_submit_batch"}
        )
        
        # Check that the batch was created successfully
        self.assertIsNotNone(batch)
        self.assertIsNotNone(batch.batch_id)
        self.assertIsNotNone(batch.provider_batch_id)
        self.assertEqual(batch.status, GenericBatchStatus.PROCESSING)
        
        # Wait for the batch to complete (with timeout)
        timeout = 300  # 5 minutes
        start_time = time.time()
        while batch.status == GenericBatchStatus.PROCESSING:
            if time.time() - start_time > timeout:
                self.fail("Batch processing timed out")
            
            # Wait a bit before checking again
            time.sleep(10)
            
            # Check batch status
            batch = await self.processor.get_batch_status(batch)
        
        # Check that the batch completed successfully
        self.assertEqual(batch.status, GenericBatchStatus.COMPLETE)
        
        # Fetch the results
        responses = await self.processor.fetch_batch_results(batch)
        
        # Check that we got the expected number of responses
        self.assertEqual(len(responses), len(requests))
        
        # Check that each response has the expected format
        for response in responses:
            self.assertIn("response", response)
            self.assertTrue(response["response"])
            
            # Check that the response contains relevant information
            self.assertIn("Bedrock", response["response"])
            self.assertIn("AWS", response["response"])

    @pytest.mark.asyncio
    async def test_multiple_models_batch(self):
        """Test batch processing with multiple models."""
        # Skip if we don't have access to Llama model
        if not check_bedrock_access(DEFAULT_LLAMA_MODEL):
            self.skipTest(f"No access to {DEFAULT_LLAMA_MODEL}")
        
        # Create requests for different models
        requests = [
            # Claude request
            GenericRequest(
                model=DEFAULT_CLAUDE_MODEL,
                prompt="What is AWS Bedrock?",
                generation_params={"temperature": 0.7, "max_tokens": 100},
                original_row={"prompt": "What is AWS Bedrock?"},
                original_row_idx=0,
                task_id=1
            ),
            # Llama request
            GenericRequest(
                model=DEFAULT_LLAMA_MODEL,
                prompt="What is AWS Bedrock?",
                generation_params={"temperature": 0.7, "max_tokens": 100},
                original_row={"prompt": "What is AWS Bedrock?"},
                original_row_idx=1,
                task_id=2
            )
        ]
        
        # Process each request with its own processor
        results = []
        for req in requests:
            # Create a processor for this model
            config = BatchRequestProcessorConfig(
                model=req.model,
                generation_params=req.generation_params,
                working_dir=self.temp_dir.name
            )
            
            processor = BedrockBatchRequestProcessor(
                config=config,
                region_name=self.region_name,
                s3_bucket=self.s3_bucket,
                s3_prefix=f"{self.s3_prefix}/{req.model}",
                role_arn=self.role_arn
            )
            
            # Convert to API-specific format
            api_request = processor.create_api_specific_request_batch(req)
            
            # Submit the batch
            batch = await processor.submit_batch(
                requests=[api_request],
                metadata={"test_id": "test_multiple_models_batch", "model": req.model}
            )
            
            # Wait for the batch to complete (with timeout)
            timeout = 300  # 5 minutes
            start_time = time.time()
            while batch.status == GenericBatchStatus.PROCESSING:
                if time.time() - start_time > timeout:
                    self.fail(f"Batch processing timed out for model {req.model}")
                
                # Wait a bit before checking again
                time.sleep(10)
                
                # Check batch status
                batch = await processor.get_batch_status(batch)
            
            # Check that the batch completed successfully
            self.assertEqual(batch.status, GenericBatchStatus.COMPLETE)
            
            # Fetch the results
            responses = await processor.fetch_batch_results(batch)
            
            # Add to results
            results.extend(responses)
        
        # Check that we got the expected number of responses
        self.assertEqual(len(results), len(requests))
        
        # Check that each response has the expected format
        for response in results:
            self.assertIn("response", response)
            self.assertTrue(response["response"])
            
            # Check that the response contains relevant information
            self.assertIn("Bedrock", response["response"])
            self.assertIn("AWS", response["response"])

    def test_llm_integration(self):
        """Test integration with the LLM class for batch processing."""
        # Create an LLM instance
        llm = LLM(
            model=self.model_id,
            provider="bedrock",
            region_name=self.region_name,
            batch=True,
            batch_params={
                "s3_bucket": self.s3_bucket,
                "s3_prefix": self.s3_prefix,
                "role_arn": self.role_arn
            }
        )
        
        # Create a list of prompts
        prompts = [
            "What is AWS Bedrock?",
            "What are foundation models?",
            "How does batch processing work in AWS Bedrock?"
        ]
        
        # Generate responses in batch mode
        responses = llm.generate_batch(prompts)
        
        # Check that we got the expected number of responses
        self.assertEqual(len(responses), len(prompts))
        
        # Check that each response has the expected format
        for response in responses:
            self.assertTrue(response)
            
            # Check that the response contains relevant information
            if "Bedrock" in response:
                self.assertIn("AWS", response)
            elif "foundation models" in response:
                self.assertIn("AI", response)
            elif "batch processing" in response:
                self.assertIn("AWS", response)
