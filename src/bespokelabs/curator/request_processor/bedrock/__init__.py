"""Bedrock request processors for batch and online inference."""

from bespokelabs.curator.request_processor.bedrock.bedrock_batch_request_processor import BedrockBatchRequestProcessor
from bespokelabs.curator.request_processor.bedrock.bedrock_online_request_processor import BedrockOnlineRequestProcessor

__all__ = ["BedrockOnlineRequestProcessor", "BedrockBatchRequestProcessor"]
