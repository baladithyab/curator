"""Integration tests for AWS Bedrock online processing.

These tests verify that the Bedrock online processor works correctly with
various models and configurations. They require valid AWS credentials
with access to Bedrock.
"""

import os
import json
import unittest
from typing import Dict, List, Any, Optional

import boto3
import pytest

from bespokelabs import curator
from bespokelabs.curator.llm.llm import LLM
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.bedrock_online_request_processor import BedrockOnlineRequestProcessor
from bespokelabs.curator.types.generic_request import GenericRequest

from tests.integrations.bedrock.utils import (
    check_aws_credentials,
    check_bedrock_access,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_LLAMA_MODEL
)


# Skip all tests if AWS credentials are not available
skip_if_no_aws_credentials = pytest.mark.skipif(
    not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")),
    reason="AWS credentials not available"
)


@skip_if_no_aws_credentials
class TestBedrockOnlineIntegration(unittest.TestCase):
    """Integration tests for the BedrockOnlineRequestProcessor."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Use a Claude model for testing
        self.model_id = os.environ.get("BEDROCK_TEST_MODEL", DEFAULT_CLAUDE_MODEL)
        self.region_name = os.environ.get("AWS_REGION", "us-east-1")
        
        # Check if we have access to Bedrock
        if not check_bedrock_access(self.model_id):
            self.skipTest(f"No access to {self.model_id}")
        
        # Create a config
        self.config = OnlineRequestProcessorConfig(
            model=self.model_id,
            generation_params={"temperature": 0.7, "max_tokens": 100}
        )
        
        # Create the processor
        self.processor = BedrockOnlineRequestProcessor(
            config=self.config,
            region_name=self.region_name
        )

    def test_create_api_specific_request_online(self):
        """Test creating API-specific request for online inference."""
        # Create a generic request
        generic_request = GenericRequest(
            model=self.model_id,
            prompt="What is AWS Bedrock?",
            generation_params={"temperature": 0.7, "max_tokens": 100}
        )
        
        # Convert to API-specific format
        api_request = self.processor.create_api_specific_request_online(generic_request)
        
        # Check that the request has the expected format
        if "anthropic" in self.model_id:
            self.assertIn("anthropic_version", api_request)
            self.assertIn("messages", api_request)
        elif "meta" in self.model_id:
            self.assertIn("prompt", api_request)
        elif "amazon" in self.model_id:
            self.assertIn("inputText", api_request)

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

    def test_multiple_models(self):
        """Test using multiple models."""
        # Skip if we don't have access to Llama model
        if not check_bedrock_access(DEFAULT_LLAMA_MODEL):
            self.skipTest(f"No access to {DEFAULT_LLAMA_MODEL}")
            
        # Skip if we're in CI environment
        if os.environ.get("CI"):
            self.skipTest("Skipping in CI environment")
            
        # Create LLM instances for different models
        claude_llm = LLM(
            model=DEFAULT_CLAUDE_MODEL,
            provider="bedrock",
            region_name=self.region_name
        )
        
        llama_llm = LLM(
            model=DEFAULT_LLAMA_MODEL,
            provider="bedrock",
            region_name=self.region_name
        )
        
        # Generate responses
        claude_response = claude_llm.generate("What is AWS Bedrock?")
        llama_response = llama_llm.generate("What is AWS Bedrock?")
        
        # Check that both responses are not empty
        self.assertTrue(claude_response)
        self.assertTrue(llama_response)
        
        # Check that both responses contain relevant information
        self.assertIn("Bedrock", claude_response)
        self.assertIn("AWS", claude_response)
        self.assertIn("Bedrock", llama_response)
        self.assertIn("AWS", llama_response)
        
        # Responses should be different
        self.assertNotEqual(claude_response, llama_response)

    def test_inference_profile(self):
        """Test using inference profiles."""
        # Skip if we're in CI environment
        if os.environ.get("CI"):
            self.skipTest("Skipping in CI environment")
            
        # Create a processor with inference profile enabled
        config = OnlineRequestProcessorConfig(
            model=self.model_id,
            generation_params={"temperature": 0.7, "max_tokens": 100}
        )
        
        processor = BedrockOnlineRequestProcessor(
            config=config,
            region_name=self.region_name,
            use_inference_profile=True
        )
        
        # Generate a response
        response = processor.generate("What is AWS Bedrock?")
        
        # Check that the response is not empty
        self.assertIn("response", response)
        self.assertTrue(response["response"])
        
        # Check that the response contains relevant information
        self.assertIn("Bedrock", response["response"])
        self.assertIn("AWS", response["response"])
