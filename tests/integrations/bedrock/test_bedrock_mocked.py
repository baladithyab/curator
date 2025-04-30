"""Unit tests for AWS Bedrock integration with mocked AWS services.

These tests verify that the Bedrock processors work correctly without
requiring actual AWS credentials or making real API calls.
"""

import unittest
from unittest.mock import patch, MagicMock

from bespokelabs.curator.llm.llm import LLM
from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online.bedrock_online_request_processor import BedrockOnlineRequestProcessor
from bespokelabs.curator.types.generic_request import GenericRequest


class TestBedrockOnlineMocked(unittest.TestCase):
    """Unit tests for the BedrockOnlineRequestProcessor with mocked AWS services."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Use a Claude model for testing
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        self.region_name = "us-east-1"

        # Create a config
        self.config = OnlineRequestProcessorConfig(
            model=self.model_id,
            generation_params={"temperature": 0.7, "max_tokens": 100}
        )

    @patch('boto3.client')
    def test_create_api_specific_request_online(self, mock_boto3_client):
        """Test creating API-specific request for online inference."""
        # Set up the mock
        mock_bedrock_runtime = MagicMock()
        mock_boto3_client.return_value = mock_bedrock_runtime

        # Create the processor with the mock
        processor = BedrockOnlineRequestProcessor(
            config=self.config,
            region_name=self.region_name
        )

        # Create a generic request
        generic_request = GenericRequest(
            model=self.model_id,
            messages=[{"role": "user", "content": "What is AWS Bedrock?"}],
            generation_params={"temperature": 0.7, "max_tokens": 100},
            original_row={"prompt": "What is AWS Bedrock?"},
            original_row_idx=0
        )

        # Convert to API-specific format
        api_request = processor.create_api_specific_request_online(generic_request)

        # Check that the request has the expected format
        self.assertIn("messages", api_request)
        self.assertIn("inferenceConfig", api_request)

        # Check that the messages are formatted correctly
        self.assertEqual(len(api_request["messages"]), 1)
        self.assertEqual(api_request["messages"][0]["role"], "user")
        self.assertEqual(api_request["messages"][0]["content"][0]["text"], "What is AWS Bedrock?")

    @patch('boto3.client')
    def test_generate_simple_prompt(self, mock_boto3_client):
        """Test generating a response for a simple prompt."""
        # Set up the mock
        mock_bedrock_runtime = MagicMock()
        mock_boto3_client.return_value = mock_bedrock_runtime

        # Create a mock for the BedrockOnlineRequestProcessor.generate method
        def mock_generate(*_args, **_kwargs):
            return {
                "response": "AWS Bedrock is a fully managed service that offers a choice of high-performing foundation models from leading AI companies.",
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            }

        # Patch the BedrockOnlineRequestProcessor.generate method
        with patch('bespokelabs.curator.request_processor.online.bedrock_online_request_processor.BedrockOnlineRequestProcessor.generate',
                  side_effect=mock_generate):
            # Create an LLM instance
            llm = LLM(
                model_name=self.model_id,
                backend="bedrock"
            )

            # Generate a response
            response = llm("What is AWS Bedrock?")

            # Check that the response is not empty
            self.assertTrue(response[0]["response"])

            # Check that the response contains the expected text
            self.assertEqual(
                response[0]["response"],
                "AWS Bedrock is a fully managed service that offers a choice of high-performing foundation models from leading AI companies."
            )

    @patch('boto3.client')
    def test_generate_with_chat_messages(self, mock_boto3_client):
        """Test generating a response with chat messages."""
        # Set up the mock
        mock_bedrock_runtime = MagicMock()
        mock_boto3_client.return_value = mock_bedrock_runtime

        # Create a mock for the BedrockOnlineRequestProcessor.generate method
        def mock_generate(*_args, **_kwargs):
            return {
                "response": "AWS Bedrock is a fully managed service that provides access to foundation models from leading AI companies.",
                "token_usage": {
                    "input_tokens": 20,
                    "output_tokens": 25
                }
            }

        # Patch the BedrockOnlineRequestProcessor.generate method
        with patch('bespokelabs.curator.request_processor.online.bedrock_online_request_processor.BedrockOnlineRequestProcessor.generate',
                  side_effect=mock_generate):
            # Create an LLM instance
            llm = LLM(
                model_name=self.model_id,
                backend="bedrock"
            )

            # Generate a response
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
                {"role": "user", "content": "What is AWS Bedrock?"}
            ]
            response = llm([messages])

            # Check that the response is not empty
            self.assertTrue(response[0]["response"])

            # Check that the response contains the expected text
            self.assertEqual(
                response[0]["response"],
                "AWS Bedrock is a fully managed service that provides access to foundation models from leading AI companies."
            )

    @patch('boto3.client')
    def test_llm_integration(self, mock_boto3_client):
        """Test integration with the LLM class."""
        # Set up the mock
        mock_bedrock_runtime = MagicMock()
        mock_boto3_client.return_value = mock_bedrock_runtime

        # Mock the converse method
        mock_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "AWS Bedrock is a fully managed service that makes foundation models available through an API."
                        }
                    ]
                }
            },
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        }
        mock_bedrock_runtime.converse.return_value = mock_response

        # Create a mock for the BedrockOnlineRequestProcessor.generate method
        def mock_generate(*_args, **_kwargs):
            return {
                "response": "AWS Bedrock is a fully managed service that makes foundation models available through an API.",
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            }

        # Patch the BedrockOnlineRequestProcessor.generate method
        with patch('bespokelabs.curator.request_processor.online.bedrock_online_request_processor.BedrockOnlineRequestProcessor.generate',
                  side_effect=mock_generate):
            # Create an LLM instance
            llm = LLM(
                model_name=self.model_id,
                backend="bedrock"
            )

            # Generate a response
            response = llm("What is AWS Bedrock?")

            # Check that the response is not empty
            self.assertTrue(response[0]["response"])

            # Check that the response contains the expected text
            self.assertEqual(
                response[0]["response"],
                "AWS Bedrock is a fully managed service that makes foundation models available through an API."
            )


if __name__ == "__main__":
    unittest.main()
