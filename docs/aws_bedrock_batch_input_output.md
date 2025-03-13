# AWS Bedrock Batch Inference: Input and Output Formats

This document provides detailed information about the input and output formats required for running batch inference jobs with AWS Bedrock. The information is based on the [official AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-data.html).

## Batch Inference Input Format

When preparing data for AWS Bedrock batch inference, you must follow these formatting requirements:

### JSONL Input Format

Your input data must be formatted as JSONL (JSON Lines) files, where each line contains a single JSON object with the following structure:

```json
{ "recordId": "11CharAlphaNum", "modelInput": {JSON body} }
```

Where:
- `recordId`: An 11-character alphanumeric string that uniquely identifies the record (optional - AWS Bedrock will add it if omitted)
- `modelInput`: A JSON object containing the model input, which must match the format of the `body` parameter used in the corresponding `InvokeModel` request for online inference

### Input Format by Model Provider

#### Anthropic Claude Models

```json
{
  "recordId": "record001",
  "modelInput": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Summarize the benefits of cloud computing in three bullet points."
          }
        ]
      }
    ],
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 300,
    "temperature": 0.7
  }
}
```

#### Amazon Titan Text Models

```json
{
  "recordId": "record002",
  "modelInput": {
    "inputText": "Write a paragraph about artificial intelligence.",
    "textGenerationConfig": {
      "maxTokenCount": 300,
      "temperature": 0.7,
      "topP": 0.9
    }
  }
}
```

#### Meta Llama Models

```json
{
  "recordId": "record003",
  "modelInput": {
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
      }
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_gen_len": 300
  }
}
```

#### Mistral AI Models

```json
{
  "recordId": "record004",
  "modelInput": {
    "messages": [
      {
        "role": "user",
        "content": "Describe three applications of machine learning in healthcare."
      }
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 300
  }
}
```

### Input Quotas and Limitations

AWS Bedrock enforces several quotas for batch inference jobs:

- **Minimum number of records per batch inference job**: A minimum number of records across all JSONL files
- **Records per input file per batch inference job**: Maximum records in a single JSONL file
- **Records per batch inference job**: Maximum total records across all JSONL files
- **Batch inference input file size**: Maximum size of a single file
- **Batch inference job size**: Maximum cumulative size of all input files

For the most current quota values, check the [AWS Service Quotas console](https://console.aws.amazon.com/servicequotas/).

## Batch Inference Output Format

After a batch inference job completes, AWS Bedrock generates the following output:

### Output JSONL Files

For each input JSONL file, AWS Bedrock generates a corresponding output JSONL file with the following structure:

```json
{ "recordId": "11CharAlphaNum", "modelInput": {JSON body}, "modelOutput": {JSON body} }
```

If there was an error processing a record, an `error` object replaces the `modelOutput` field:

```json
{ "recordId": "11CharAlphaNum", "modelInput": {JSON body}, "error": {"errorCode": 400, "errorMessage": "bad request"} }
```

### Output Format by Model Provider

#### Anthropic Claude Models

```json
{
  "recordId": "record001",
  "modelInput": { ... },
  "modelOutput": {
    "content": [
      {
        "type": "text",
        "text": "Here are three benefits of cloud computing:\n\n• Scalability: Easily scale resources up or down based on demand\n• Cost-efficiency: Pay only for what you use without major upfront investments\n• Accessibility: Access your data and applications from anywhere with an internet connection"
      }
    ],
    "usage": {
      "input_tokens": 15,
      "output_tokens": 57
    }
  }
}
```

#### Amazon Titan Text Models

```json
{
  "recordId": "record002",
  "modelInput": { ... },
  "modelOutput": {
    "inputTextTokenCount": 7,
    "results": [
      {
        "tokenCount": 62,
        "outputText": "Artificial intelligence (AI) represents a frontier of technological advancement that seeks to create machines capable of mimicking human cognitive functions...",
        "completionReason": "FINISH"
      }
    ]
  }
}
```

#### Meta Llama Models

```json
{
  "recordId": "record003",
  "modelInput": { ... },
  "modelOutput": {
    "generation": "Quantum computing is like traditional computing but uses quantum bits or 'qubits' instead of regular bits...",
    "prompt_token_count": 8,
    "generation_token_count": 65
  }
}
```

#### Mistral AI Models

```json
{
  "recordId": "record004",
  "modelInput": { ... },
  "modelOutput": {
    "choices": [
      {
        "message": {
          "content": "Here are three applications of machine learning in healthcare:\n\n1. Disease diagnosis and prediction...",
          "role": "assistant"
        }
      }
    ],
    "usage": {
      "prompt_tokens": 13,
      "completion_tokens": 78
    }
  }
}
```

### Manifest File

AWS Bedrock also generates a `manifest.json.out` file containing summary statistics:

```json
{
  "totalRecordCount": 100,
  "processedRecordCount": 100,
  "successRecordCount": 98,
  "errorRecordCount": 2,
  "inputTokenCount": 1250,
  "outputTokenCount": 6430
}
```

The fields in the manifest file are:
- `totalRecordCount`: Total number of records submitted to the batch job
- `processedRecordCount`: Number of records processed in the batch job
- `successRecordCount`: Number of records successfully processed
- `errorRecordCount`: Number of records that caused errors
- `inputTokenCount`: Total number of input tokens submitted
- `outputTokenCount`: Total number of output tokens generated

## Best Practices

1. **Validate Input Format**: Ensure your JSONL input files match the required format for the specific model you're using
2. **Set Appropriate Parameters**: Configure appropriate generation parameters (temperature, max_tokens, etc.) based on your use case
3. **Monitor Job Status**: Use the AWS console or API to monitor the status of your batch jobs
4. **Handle Errors**: Check the manifest file and error records to identify and address any issues
5. **Consider Regional Availability**: Ensure your chosen model supports batch inference in your target region

## Related Documentation

- [AWS Bedrock Integration](aws_bedrock_integration.md)
- [AWS Bedrock Batch Model Availability](aws_bedrock_batch_model_availability.md)
- [AWS Bedrock Service Quotas](aws_bedrock_service_quotas.md)
- [AWS Bedrock Capabilities](aws_bedrock_capabilities.md)

## Additional Resources

For the most up-to-date information on AWS Bedrock batch inference, refer to the following official AWS documentation:
- [Format and upload your batch inference data](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-data.html)
- [View the results of a batch inference job](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-results.html)
- [Create a batch inference job](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-create.html) 