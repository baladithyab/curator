"""Utility functions for Bedrock test scripts.

This module provides common utility functions used across Bedrock test scripts,
including verbosity control and result formatting.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Global verbosity setting
VERBOSE = os.environ.get("BEDROCK_TEST_VERBOSE", "0").lower() in ("1", "true", "yes", "y")

def is_verbose() -> bool:
    """Check if verbose output is enabled."""
    return VERBOSE

def vprint(*args, **kwargs) -> None:
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)

def print_header(message: str) -> None:
    """Print a header message."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def print_subheader(message: str) -> None:
    """Print a subheader message."""
    print("\n" + "-" * 60)
    print(f" {message}")
    print("-" * 60)

def format_response_summary(prompt: str, response: Any, token_usage: Optional[Dict] = None) -> str:
    """Format a response summary for display.
    
    Args:
        prompt: The original prompt
        response: The model response
        token_usage: Optional token usage information
        
    Returns:
        A formatted string with the response summary
    """
    # Extract the response text
    if isinstance(response, dict):
        response_text = response.get("response", str(response))
        usage = response.get("token_usage", token_usage)
    elif hasattr(response, 'response'):
        response_text = response.response
        usage = getattr(response, 'token_usage', token_usage)
    else:
        response_text = str(response)
        usage = token_usage
    
    # Truncate long responses for summary view
    if len(response_text) > 150:
        response_text = response_text[:147] + "..."
    
    # Format the summary
    summary = f"Prompt: {prompt}\nResponse: {response_text}"
    
    # Add token usage if available
    if usage:
        summary += f"\nToken usage: {usage}"
    
    return summary

def print_test_result(test_name: str, success: bool, error: Optional[Exception] = None) -> None:
    """Print the result of a test.
    
    Args:
        test_name: Name of the test
        success: Whether the test was successful
        error: Optional exception if the test failed
    """
    if success:
        print(f"✅ {test_name}: PASSED")
    else:
        print(f"❌ {test_name}: FAILED")
        if error:
            print(f"   Error: {str(error)}")

def time_execution(func):
    """Decorator to time the execution of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        vprint(f"Execution time for {func.__name__}: {end_time - start_time:.2f}s")
        return result
    return wrapper

def check_aws_environment():
    """Check if AWS credentials are properly configured."""
    try:
        # Test if we can access AWS credentials (including role-based credentials)
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        vprint(f"Using AWS credentials for: {identity['Arn']}")
        return True, identity['Arn']
    except Exception as e:
        print(f"Error accessing AWS credentials: {str(e)}")
        print("Make sure you have valid AWS credentials configured.")
        return False, None
