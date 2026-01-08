# Bedrock Quota Management Improvements

## Executive Summary

This document analyzes the current implementation of AWS Bedrock quota management in the Curator request processor and provides recommendations for improvements. The goal is to ensure robust, accurate, and dynamic quota handling to maximize throughput while avoiding throttling errors, particularly for new features like Cross-Region Inference.

## Current Implementation Analysis

The current implementation resides primarily in `src/bespokelabs/curator/request_processor/bedrock/bedrock_availability.py`.

### Strengths
-   **Hybrid Approach**: Uses a combination of dynamic discovery via AWS Service Quotas API and static fallbacks for resilience.
-   **Model Awareness**: Maintains a cache of model capabilities (modalities, API support).
-   **Lazy Initialization**: Clients are initialized only when needed.

### Weaknesses & Gaps
1.  **Fragile Quota Matching**: Relies on string matching of "friendly names" against Service Quota names (e.g., "Anthropic Claude 3 Haiku"). This is prone to breakage if AWS changes naming conventions or for new models with unexpected names.
2.  **Cross-Region Inference (CRI) Gaps**: The current logic explicitly looks for "On-demand model inference" in quota names. CRI quotas typically use different naming (e.g., "Cross-Region InvokeModel..."), which means CRI profiles likely fall back to conservative static defaults.
3.  **Static Fallback Maintenance**: The `_STATIC_MODEL_QUOTAS` dictionary requires manual updates for every new model release.
4.  **Lack of Adaptive Rate Limiting**: While rate limit errors are tracked, the system does not appear to automatically lower the effective rate limit (RPM/TPM) in real-time when throttling occurs. It relies on the initial quota fetch.

## AWS Bedrock Quota System

### Service Quotas API
Bedrock quotas are managed via the AWS Service Quotas API (`service-quotas`).
-   **Service Code**: `bedrock`
-   **Retrieval**: `list_service_quotas` returns a list of all quotas for the service.
-   **Identification**: Quotas are identified by `QuotaCode` (immutable ID) and `QuotaName` (human-readable).

### Quota Naming Conventions
-   **On-Demand**: "On-demand model inference requests per minute for [Model Name]"
-   **Cross-Region**: "Cross-Region InvokeModel requests per minute for [Inference Profile Name]" (Note: Exact wording needs to be handled flexibly).
-   **Batch**: "Batch inference job..."

### Cross-Region Inference Profiles
CRI profiles (e.g., `us.anthropic.claude-3-5-sonnet-20240620-v1:0`) have distinct quotas separate from the base regional model quotas. These must be queried specifically.

## Recommendations

### 1. Robust Quota Matching Logic
Update `get_model_quotas` in `bedrock_availability.py` to handle different quota types more flexibly.

**Proposed Logic:**
```python
# Pseudocode
if is_cross_region_profile(model_id):
    required_terms = ["cross-region", "invokemodel"]
else:
    required_terms = ["on-demand", "model inference"]

# Filter quotas that contain all required terms AND the model name
```

### 2. Support for Cross-Region Inference Quotas
Explicitly add support for parsing CRI quotas. The current implementation likely misses them because it filters for "on-demand".

### 3. Adaptive Rate Limiting
Implement a "backoff and clamp" mechanism in `BedrockOnlineRequestProcessor`.
-   **Trigger**: When a `ThrottlingException` occurs.
-   **Action**:
    1.  Temporarily reduce the effective RPM/TPM (e.g., to 80% of current).
    2.  Update the `OnlineStatusTracker` or a shared state to reflect this lower limit.
    3.  Periodically attempt to recover (slow start) or refresh quotas from API.

### 4. Enhanced Logging for Unmatched Models
Add warning logs when a model is found in `list_foundation_models` but no corresponding quota is found in `list_service_quotas`. This will alert developers to name mismatches or missing static mappings.

### 5. Batch Quota Utilization
Ensure that `BedrockBatchRequestProcessor` actively checks `BedrockQuotas` before submitting jobs to avoid `LimitExceededException` on job submission.

## Implementation Plan

1.  **Refactor `bedrock_availability.py`**:
    -   Update `_get_model_friendly_name` to support CRI profile names.
    -   Update `get_model_quotas` to search for "Cross-Region" quotas when appropriate.
    -   Improve string matching to be case-insensitive and more robust.

2.  **Update Static Fallbacks**:
    -   Add entries for recent models (Claude 3.5 Sonnet v2, Nova, etc.) if missing.
    -   Add default quotas for CRI prefixes (e.g., `us.`, `eu.`).

3.  **Verify & Test**:
    -   Unit tests for quota matching logic.
    -   Integration test (if possible) or manual verification against a live AWS account with CRI enabled.
