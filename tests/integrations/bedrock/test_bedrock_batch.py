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

    def _generate_batch_test_data(self, num_entries: int, model_id: str) -> List[GenericRequest]:
        """Helper to generate a list of GenericRequest objects for testing."""
        requests = []
        for i in range(num_entries):
            prompt = f"This is test prompt number {i} for Bedrock batch processing. What is 2+2?"
            requests.append(
                GenericRequest(
                    model=model_id,
                    prompt=prompt,
                    generation_params={"temperature": 0.7, "max_tokens": 50},
                    original_row={"prompt": prompt, "index": i},
                    original_row_idx=i,
                    task_id=str(i) # Ensure task_id is a string
                )
            )
        return requests

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
        """Test submitting a batch job to Bedrock with >100 entries and <50MB size."""
        num_test_entries = 105  # Test with more than 100 entries
        requests = self._generate_batch_test_data(num_test_entries, self.model_id)
        
        # Convert to API-specific format
        api_requests = [
            self.processor.create_api_specific_request_batch(req)
            for req in requests
        ]

        # Create a temporary file to check its size before S3 upload
        # This mimics what submit_batch would do internally for the input file.
        temp_input_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", dir=self.temp_dir.name) as tmp_f:
                for api_req in api_requests:
                    tmp_f.write(json.dumps(api_req) + "\n")
                temp_input_file_path = tmp_f.name
            
            self.assertIsNotNone(temp_input_file_path)
            file_size_bytes = os.path.getsize(temp_input_file_path)
            max_allowed_bytes = 50 * 1024 * 1024  # 50MB
            self.assertLessEqual(file_size_bytes, max_allowed_bytes,
                                 f"Generated input file size {file_size_bytes} exceeds {max_allowed_bytes} bytes.")
            print(f"Generated test input file with {num_test_entries} entries, size: {file_size_bytes} bytes.")

        finally:
            if temp_input_file_path and os.path.exists(temp_input_file_path):
                os.remove(temp_input_file_path)
        
        # Submit the batch (actual S3 upload happens here)
        batch = await self.processor.submit_batch(
            requests=api_requests,
            metadata={"test_id": "test_submit_batch_large"}
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
        self.assertEqual(len(responses), len(requests), "Number of responses did not match number of requests.")
        
        # Check that each response has the expected format
        for i, response_obj in enumerate(responses):
            # Assuming GenericResponse objects are returned
            self.assertIsInstance(response_obj, curator.types.generic_response.GenericResponse, f"Response {i} is not a GenericResponse object.")
            self.assertTrue(response_obj.success, f"Response {i} was not successful: {response_obj.message}")
            self.assertIsNotNone(response_obj.response, f"Response {i} content is None.")
            self.assertTrue(len(response_obj.response) > 0, f"Response {i} content is empty.")
            
            # Check that the response contains relevant information (e.g., part of the prompt or expected answer)
            # This part is model/prompt dependent. For "What is 2+2?", expect "4".
            if "2+2" in requests[i].prompt: # Check if it was the math prompt
                 self.assertIn("4", response_obj.response, f"Response {i} for '2+2' did not contain '4'. Got: {response_obj.response}")
            elif "Bedrock" in requests[i].prompt: # Fallback for original prompt
                 self.assertIn("Bedrock", response_obj.response, f"Response {i} for Bedrock prompt did not contain 'Bedrock'. Got: {response_obj.response}")

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
        
        # Create a list of prompts (at least 100)
        num_llm_prompts = 102
        prompts = [f"This is LLM integration test prompt {i}. Tell me a short story." for i in range(num_llm_prompts)]
        
        # Generate responses in batch mode
        responses = llm.generate_batch(prompts)
        
        # Check that we got the expected number of responses
        self.assertEqual(len(responses), len(prompts), "Number of LLM batch responses did not match number of prompts.")
        
        # Check that each response has the expected format
        for i, response_text in enumerate(responses):
            self.assertIsInstance(response_text, str, f"LLM Response {i} is not a string.")
            self.assertTrue(len(response_text) > 0, f"LLM Response {i} is empty.")
            # A simple check, could be more specific if prompts had expected keywords in answers
            self.assertIn("story", response_text.lower(), f"LLM Response {i} did not seem to contain a story. Got: {response_text[:100]}...")
