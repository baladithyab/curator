# AWS Bedrock Service Quotas and Throttling Management Guide

This guide provides comprehensive information about AWS Bedrock service quotas, how to access them, monitor usage metrics, and implement throttling-aware logic in your AWS Bedrock implementations.

## Table of Contents
1. [Understanding AWS Bedrock Service Quotas](#understanding-aws-bedrock-service-quotas)
2. [Accessing Service Quotas](#accessing-service-quotas)
3. [Monitoring Bedrock Usage with CloudWatch](#monitoring-bedrock-usage-with-cloudwatch)
4. [Implementing Throttling-Aware Logic](#implementing-throttling-aware-logic)
5. [Best Practices](#best-practices)

## Understanding AWS Bedrock Service Quotas

AWS Bedrock has several service quotas (formerly called limits) that restrict the rate at which you can make API calls or the number of resources you can create. These quotas help ensure service stability and protect your account from unexpected charges.

Key quotas include:
- Maximum transactions per second (TPS) for model inference
- Maximum concurrent batch inference jobs per model
- Maximum tokens per request (input + output)
- Maximum batch inference jobs per region

These quotas may vary by:
- AWS Region
- Model provider (Anthropic, Amazon, AI21, etc.)
- Model type (Claude, Titan, etc.)
- Account type and usage history

## Accessing Service Quotas

### Via AWS Management Console

1. Navigate to the AWS Service Quotas console (https://console.aws.amazon.com/servicequotas/)
2. Select "AWS services" from the left navigation pane
3. Search for "Amazon Bedrock" and select it
4. View your current quotas and request increases if needed

### Via AWS CLI

List all service quotas for AWS Bedrock:

```bash
aws service-quotas list-service-quotas --service-code bedrock
```

Get details for a specific quota:

```bash
aws service-quotas get-service-quota \
    --service-code bedrock \
    --quota-code [QUOTA-CODE]
```

Example output:
```json
{
    "Quota": {
        "ServiceCode": "bedrock",
        "ServiceName": "Amazon Bedrock",
        "QuotaArn": "arn:aws:servicequotas:us-west-2:123456789012:bedrock/L-1234567",
        "QuotaCode": "L-1234567",
        "QuotaName": "Model inference requests per second for Anthropic Claude 3 Opus",
        "Value": 5.0,
        "Unit": "None",
        "Adjustable": true,
        "GlobalQuota": false
    }
}
```

### Via Python SDK (boto3)

```python
import boto3

def get_bedrock_service_quotas():
    """Retrieve all service quotas for AWS Bedrock."""
    client = boto3.client('service-quotas')
    
    paginator = client.get_paginator('list_service_quotas')
    quota_pages = paginator.paginate(ServiceCode='bedrock')
    
    quotas = []
    for page in quota_pages:
        quotas.extend(page['Quotas'])
    
    return quotas

def get_specific_quota(quota_code):
    """Get a specific quota by its code."""
    client = boto3.client('service-quotas')
    
    response = client.get_service_quota(
        ServiceCode='bedrock',
        QuotaCode=quota_code
    )
    
    return response['Quota']
```

## Monitoring Bedrock Usage with CloudWatch

AWS Bedrock automatically sends metrics to Amazon CloudWatch, which you can use to monitor your usage and set up alarms.

### Key Metrics to Monitor

| Metric Name | Description | Use Case |
|-------------|-------------|----------|
| `Invocations` | Number of requests to InvokeModel or other API operations | Track overall usage |
| `InvocationLatency` | Latency of invocations in milliseconds | Monitor performance |
| `InvocationThrottles` | Number of throttled requests | Detect quota limits being hit |
| `InputTokenCount` | Number of input tokens | Track token usage for billing |
| `OutputTokenCount` | Number of output tokens | Track token usage for billing |
| `InvocationClientErrors` | Number of client-side errors | Detect issues with requests |
| `InvocationServerErrors` | Number of server-side errors | Detect AWS service issues |

### Setting Up CloudWatch Alarms

Create an alarm to notify you when you approach service quota limits:

```python
import boto3

def create_throttling_alarm():
    """Create a CloudWatch alarm for throttling events."""
    cloudwatch = boto3.client('cloudwatch')
    
    response = cloudwatch.put_metric_alarm(
        AlarmName='BedrockThrottlingAlarm',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=1,
        MetricName='InvocationThrottles',
        Namespace='AWS/Bedrock',
        Period=60,  # 1 minute
        Statistic='Sum',
        Threshold=5,  # Alert if more than 5 throttling events in a minute
        ActionsEnabled=True,
        AlarmDescription='Alarm when Bedrock API calls are being throttled',
        AlarmActions=['arn:aws:sns:region:account-id:alarm-topic'],
        Dimensions=[
            {
                'Name': 'ModelId',
                'Value': 'anthropic.claude-3-opus-20240229-v1:0'
            }
        ]
    )
    
    return response
```

## Implementing Throttling-Aware Logic

To handle throttling gracefully in your applications, implement retry logic with exponential backoff.

### Python Example with Boto3

```python
import boto3
import time
import random
from botocore.exceptions import ClientError

def invoke_bedrock_with_retry(model_id, body, max_retries=5, initial_backoff=1):
    """
    Invoke Bedrock model with retry logic for throttling exceptions.
    
    Args:
        model_id: The model ID to use
        body: The request body (JSON)
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        
    Returns:
        The model response or raises an exception after max retries
    """
    bedrock_runtime = boto3.client('bedrock-runtime')
    
    retries = 0
    while True:
        try:
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            return response
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            
            # Only retry on throttling exceptions
            if error_code in ['ThrottlingException', 'ServiceQuotaExceededException']:
                if retries >= max_retries:
                    raise Exception(f"Maximum retries reached ({max_retries}). Last error: {str(e)}")
                
                # Calculate backoff with jitter
                backoff = initial_backoff * (2 ** retries) + random.uniform(0, 1)
                print(f"Request throttled. Retrying in {backoff:.2f} seconds (attempt {retries+1}/{max_retries})")
                time.sleep(backoff)
                retries += 1
            else:
                # For non-throttling errors, raise immediately
                raise
```

### Implementing a Rate Limiter

For more advanced control, implement a client-side rate limiter:

```python
import time
import threading

class BedrockRateLimiter:
    """Rate limiter for AWS Bedrock API calls."""
    
    def __init__(self, max_tps=2):
        """
        Initialize the rate limiter.
        
        Args:
            max_tps: Maximum transactions per second allowed
        """
        self.max_tps = max_tps
        self.tokens = max_tps
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
    
    def acquire(self):
        """
        Acquire a token for making an API call.
        Blocks until a token is available.
        """
        with self.lock:
            current_time = time.time()
            time_since_refill = current_time - self.last_refill_time
            
            # Refill tokens based on elapsed time
            new_tokens = time_since_refill * self.max_tps
            self.tokens = min(self.max_tps, self.tokens + new_tokens)
            self.last_refill_time = current_time
            
            if self.tokens < 1:
                # Calculate wait time until next token is available
                wait_time = (1 - self.tokens) / self.max_tps
                return wait_time
            
            # Consume a token
            self.tokens -= 1
            return 0
    
    def execute(self, func, *args, **kwargs):
        """
        Execute a function with rate limiting.
        
        Args:
            func: The function to execute
            *args, **kwargs: Arguments to pass to func
            
        Returns:
            The result of func(*args, **kwargs)
        """
        wait_time = self.acquire()
        if wait_time > 0:
            time.sleep(wait_time)
        return func(*args, **kwargs)
```

Usage example:

```python
# Create a rate limiter for Bedrock API calls
rate_limiter = BedrockRateLimiter(max_tps=5)  # Allow 5 TPS

# Use the rate limiter to execute API calls
def invoke_model(model_id, input_text):
    bedrock_runtime = boto3.client('bedrock-runtime')
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps({"prompt": input_text})
    )
    return response

# This will respect the rate limit
result = rate_limiter.execute(invoke_model, "anthropic.claude-3-opus-20240229-v1:0", "Hello Claude!")
```

## Dynamically Adapting to Service Quotas

To create a fully dynamic solution that adapts to your current service quotas:

```python
import boto3
import json

class AdaptiveBedrockClient:
    """A Bedrock client that adapts to service quotas."""
    
    def __init__(self):
        self.service_quotas = boto3.client('service-quotas')
        self.bedrock_runtime = boto3.client('bedrock-runtime')
        self.cloudwatch = boto3.client('cloudwatch')
        self.quota_cache = {}
        self.rate_limiters = {}
    
    def get_model_quota(self, model_id):
        """Get the TPS quota for a specific model."""
        # First check cache
        if model_id in self.quota_cache:
            return self.quota_cache[model_id]
        
        # Map model_id to quota code (this mapping would need to be maintained)
        model_to_quota_code = {
            "anthropic.claude-3-opus-20240229-v1:0": "L-12345678",
            "anthropic.claude-3-sonnet-20240229-v1:0": "L-23456789",
            # Add other models as needed
        }
        
        quota_code = model_to_quota_code.get(model_id)
        if not quota_code:
            # Default to a conservative value if model not found
            return 2
        
        try:
            response = self.service_quotas.get_service_quota(
                ServiceCode='bedrock',
                QuotaCode=quota_code
            )
            quota = response['Quota']['Value']
            
            # Cache the result
            self.quota_cache[model_id] = quota
            return quota
        except Exception as e:
            print(f"Error getting quota for {model_id}: {str(e)}")
            # Return a conservative default
            return 2
    
    def get_current_usage(self, model_id):
        """
        Get current usage rate from CloudWatch metrics.
        Returns requests per second over the last minute.
        """
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Bedrock',
                MetricName='Invocations',
                Dimensions=[
                    {
                        'Name': 'ModelId',
                        'Value': model_id
                    }
                ],
                StartTime=time.time() - 300,  # Last 5 minutes
                EndTime=time.time(),
                Period=60,  # 1-minute periods
                Statistics=['Sum']
            )
            
            if not response['Datapoints']:
                return 0
            
            # Get the most recent datapoint
            latest = max(response['Datapoints'], key=lambda x: x['Timestamp'])
            # Convert sum per minute to per second
            return latest['Sum'] / 60
        except Exception as e:
            print(f"Error getting usage for {model_id}: {str(e)}")
            return 0
    
    def get_rate_limiter(self, model_id):
        """Get or create a rate limiter for a model."""
        if model_id not in self.rate_limiters:
            quota = self.get_model_quota(model_id)
            # Set limit to 80% of quota to provide a safety margin
            self.rate_limiters[model_id] = BedrockRateLimiter(max_tps=quota * 0.8)
        return self.rate_limiters[model_id]
    
    def invoke_model(self, model_id, body):
        """Invoke a model with quota-aware rate limiting."""
        limiter = self.get_rate_limiter(model_id)
        
        def _call():
            return self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body) if isinstance(body, dict) else body
            )
        
        return limiter.execute(_call)
```

## Best Practices

1. **Monitor Your Usage**: Set up CloudWatch dashboards to track API calls, throttling events, and token usage.

2. **Implement Graceful Degradation**: When approaching limits, consider falling back to alternative models or reducing features.

3. **Request Quota Increases Proactively**: If you anticipate higher usage, request quota increases well in advance.

4. **Use Batch Processing**: Where possible, use AWS Bedrock Batch Inference to process multiple items efficiently.

5. **Optimize Token Usage**: Reduce unnecessary tokens in prompts to stay within limits and lower costs.

6. **Implement Client-Side Caching**: Cache responses for identical or similar requests to reduce API calls.

7. **Add Jitter to Retries**: When implementing retries, add randomness to backoff times to prevent "thundering herd" problems.

8. **Set Up Alerting**: Create CloudWatch alarms to notify you before you hit your quotas.

9. **Periodically Refresh Quota Information**: Service quotas may change over time, so refresh your cached quota values regularly.

10. **Test Throttling Scenarios**: Test how your application handles throttling events before deploying to production.

## Conclusion

Effectively managing AWS Bedrock service quotas and implementing throttling-aware logic is essential for building reliable AI applications. By monitoring your usage, implementing proper retry mechanisms, and adapting to your service quotas, you can ensure your applications remain available and resilient even under high load.

For the most up-to-date information on AWS Bedrock service quotas, refer to the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html).
