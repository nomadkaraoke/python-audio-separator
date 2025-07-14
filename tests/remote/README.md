# Remote API Tests

This directory contains comprehensive tests for the remote API functionality of audio-separator, which allows running audio separation workloads on remote servers via an API.

## Test Structure

### Unit Tests (`tests/unit/`)

- **`test_remote_api_client.py`** - Tests for the `AudioSeparatorAPIClient` class
  - Mock-based tests for all client methods
  - Tests HTTP request/response handling
  - Tests error conditions and edge cases
  - Tests the convenience method `separate_audio_and_wait()`

- **`test_remote_cli.py`** - Tests for the remote CLI functionality
  - Tests command-line argument parsing
  - Tests all CLI commands (separate, status, models, download)
  - Tests error handling and edge cases
  - Mock-based tests without actual HTTP calls

### Integration Tests (`tests/integration/`)

- **`test_remote_api_integration.py`** - Full integration tests
  - Tests with a mock HTTP server that simulates the real API
  - End-to-end workflow testing
  - Tests job submission, polling, and file download
  - Tests multiple concurrent jobs
  - Tests error handling with realistic scenarios

## Remote API Architecture

The remote API system consists of three main components:

1. **API Server** (`audio_separator/remote/deploy_modal.py`)
   - FastAPI server deployed on Modal.com
   - Handles audio upload, processing, and file serving
   - Supports multiple models in single jobs
   - Asynchronous processing with job status tracking

2. **API Client** (`audio_separator/remote/api_client.py`)
   - Python client for interacting with the remote API
   - Handles file uploads, job polling, and downloads
   - Supports all separator parameters and architectures

3. **Remote CLI** (`audio_separator/remote/cli.py`)
   - Command-line interface using the API client
   - Mirror of local CLI functionality but for remote processing
   - Supports all local CLI parameters and options

## Key Features Tested

### Multiple Model Support
- Upload once, process with multiple models
- Efficient workflow for comparing model quality
- Progress tracking across multiple models

### Full Parameter Compatibility
- All MDX, VR, Demucs, and MDXC parameters supported
- Custom output naming and format options
- Same parameter validation as local processing

### Asynchronous Processing
- Job submission returns immediately with task ID
- Status polling with progress updates
- Background processing on remote server

### Error Handling
- Network connectivity issues
- Invalid parameters and file formats
- Server errors and timeouts
- Job failures and cleanup

## Running the Tests

### Run All Remote API Tests
```bash
# Run all unit tests for remote functionality
pytest tests/unit/test_remote_api_client.py tests/unit/test_remote_cli.py -v

# Run integration tests with mock server
pytest tests/integration/test_remote_api_integration.py -v

# Run all remote tests
pytest tests/unit/test_remote*.py tests/integration/test_remote*.py -v
```

### Run Specific Test Categories
```bash
# Test only API client functionality
pytest tests/unit/test_remote_api_client.py::TestAudioSeparatorAPIClient -v

# Test only CLI functionality  
pytest tests/unit/test_remote_cli.py::TestRemoteCLI -v

# Test end-to-end workflows
pytest tests/integration/test_remote_api_integration.py::TestRemoteAPIEndToEnd -v
```

### Run with Coverage
```bash
pytest tests/unit/test_remote*.py tests/integration/test_remote*.py --cov=audio_separator.remote --cov-report=html
```

## Test Environment Setup

The tests are designed to run without requiring a live API server:

1. **Unit Tests** - Use mocked HTTP responses, no network calls
2. **Integration Tests** - Use a mock HTTP server that simulates the real API
3. **End-to-End Tests** - Full workflow testing with realistic data

### Mock Server Features

The integration tests include a comprehensive mock HTTP server that simulates:
- Job submission and processing
- Status polling with progress updates
- File upload and download
- Model listing and filtering
- Error conditions and edge cases

## Testing Best Practices

### Isolated Tests
- Each test is independent and can run in isolation
- Temporary files are properly cleaned up
- Mock state is reset between tests

### Realistic Scenarios
- Tests use realistic audio file formats and sizes
- Error conditions match real-world scenarios
- Progress tracking mimics actual processing times

### Comprehensive Coverage
- All API endpoints are tested
- All CLI commands and options are tested
- Both success and error paths are tested
- Parameter validation and edge cases are covered

## Test Data

Tests use minimal synthetic data to avoid large file dependencies:
- Fake audio content for upload testing
- Simulated processing results
- Mock model metadata

## Debugging Tests

### Enable Debug Logging
```bash
pytest tests/integration/test_remote_api_integration.py -v -s --log-cli-level=DEBUG
```

### Run Individual Tests
```bash
# Test specific functionality
pytest tests/unit/test_remote_api_client.py::TestAudioSeparatorAPIClient::test_separate_audio_and_wait_success -v -s

# Test with specific parameters
pytest tests/integration/test_remote_api_integration.py::TestRemoteAPIIntegration::test_job_status_polling -v -s
```

### Test Environment Variables
```bash
# Skip certain tests if needed
SKIP_INTEGRATION_TESTS=1 pytest tests/unit/test_remote*.py

# Enable additional debug output
DEBUG_REMOTE_TESTS=1 pytest tests/integration/test_remote*.py -v -s
```

## Integration with CI/CD

These tests are designed to run in CI environments:
- No external dependencies required
- Fast execution (typically < 30 seconds)
- Reliable mock server implementation
- Clear pass/fail criteria

## Contributing

When adding new remote API features:

1. **Add unit tests** for individual components
2. **Add integration tests** for end-to-end workflows
3. **Update mock server** to support new endpoints
4. **Test error conditions** and edge cases
5. **Update documentation** for new test scenarios

The test suite should maintain high coverage while remaining fast and reliable for continuous integration. 