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

## Related Documentation

- [AWS Bedrock Integration](aws_bedrock_integration.md)
- [AWS Bedrock Batch Model Availability](aws_bedrock_batch_model_availability.md)
- [AWS Bedrock Capabilities](aws_bedrock_capabilities.md)

## Additional Resources

For the most up-to-date information on AWS Bedrock service quotas, refer to the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html). 