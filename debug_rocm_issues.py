#!/usr/bin/env python3
"""
Debug script to diagnose ROCm implementation crashes in audio-separator.
This script tests key operations that commonly cause issues with ROCm.
"""

import os
import random
import re
import subprocess
import sys
import time
import traceback

import torch
import torch.amp


def get_rocm_info():
    """Get ROCm system information."""
    rocm_info = {}

    # Try to get ROCm version
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        rocm_info["rocminfo"] = result.stdout

        # Extract version from rocminfo output
        version_match = re.search(
            r"ROCm Stack Version: (.+)$", result.stdout, re.MULTILINE
        )
        if version_match:
            rocm_info["version"] = version_match.group(1)

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        rocm_info["rocminfo"] = "rocminfo command not found or failed"

    # Try to get HIP version
    try:
        result = subprocess.run(
            ["hipconfig"], capture_output=True, text=True, timeout=10
        )
        rocm_info["hipconfig"] = result.stdout

        # Extract HIP version
        version_match = re.search(r"HIP version\s+:\s+(.+)", result.stdout)
        if version_match:
            rocm_info["hip_version"] = version_match.group(1)

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        rocm_info["hipconfig"] = "hipconfig command not found or failed"

    return rocm_info


def get_system_info():
    """Get system information."""
    system_info = {}

    # Get CPU information
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            cpu_model = re.search(r"model name\s+: (.+)$", cpuinfo, re.MULTILINE)
            if cpu_model:
                system_info["cpu_model"] = cpu_model.group(1)

            cpu_cores = len(re.findall(r"^processor", cpuinfo, re.MULTILINE))
            system_info["cpu_cores"] = cpu_cores

    except Exception as e:
        system_info["cpu_info"] = f"Failed to read CPU info: {e}"

    # Get memory information
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
            mem_total = re.search(r"MemTotal:\s+(\d+) kB", meminfo)
            if mem_total:
                system_info["memory_total"] = (
                    int(mem_total.group(1)) / 1024
                )  # Convert to MB

    except Exception as e:
        system_info["memory_info"] = f"Failed to read memory info: {e}"

    # Get kernel version
    try:
        with open("/proc/version", "r") as f:
            system_info["kernel_version"] = f.read().strip()
    except Exception as e:
        system_info["kernel_version"] = f"Failed to read kernel version: {e}"

    return system_info


# Set environment variables for ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.2"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1030"


def print_section(title):
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")


def test_pytorch_rocm_setup():
    """Test basic PyTorch and ROCm setup."""
    print_section("1. PyTorch and ROCm Setup")

    print(f"PyTorch version: {torch.__version__}")
    print(f"ROCm detected in version: {'+rocm' in torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        # Test ROCm-specific features
        device = torch.device("cuda")

        # Test ROCm version and capabilities
        try:
            # Check for ROCm-specific attributes
            if hasattr(torch.cuda, "get_device_properties"):
                props = torch.cuda.get_device_properties(0)
                print(f"GPU Architecture: {props.major}.{props.minor}")
                print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")

            # Test ROCm version
            if hasattr(torch.version, "rocm"):
                print(f"PyTorch ROCm version: {torch.version.rocm}")

        except Exception as e:
            print(f"✗ ROCm property test failed - {type(e).__name__}: {e}")

        # Test basic tensor operations
        try:
            x = torch.randn(1000, 1000).to("cuda")
            y = torch.randn(1000, 1000).to("cuda")
            z = torch.mm(x, y)
            print("✓ Basic tensor multiplication on GPU: PASSED")
        except Exception as e:
            print(
                f"✗ Basic tensor multiplication on GPU: FAILED - {type(e).__name__}: {e}"
            )

        # Test ROCm memory operations
        try:
            # Test memory pinning
            cpu_tensor = torch.randn(1000, 1000)
            pinned_tensor = cpu_tensor.pin_memory()
            gpu_tensor = pinned_tensor.to(device, non_blocking=True)
            print("✓ Memory pinning and non-blocking transfer successful")

            # Test stream operations
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.mm(a, b)
            stream.synchronize()
            print("✓ CUDA stream operations successful")

            # Test event operations
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            d = torch.mm(c, c)
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"✓ CUDA event timing successful: {elapsed_time:.2f}ms")

        except Exception as e:
            print(f"✗ ROCm memory/stream test failed - {type(e).__name__}: {e}")

        # Test ROCm-specific math operations
        try:
            # Test half precision
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).major >= 5
            ):
                x_half = torch.randn(100, 100, device=device, dtype=torch.float16)
                y_half = torch.randn(100, 100, device=device, dtype=torch.float16)
                z_half = torch.mm(x_half, y_half)
                print("✓ Half precision (float16) operations successful")

            # Test complex number operations
            x_complex = torch.randn(100, 100, dtype=torch.complex64, device=device)
            y_complex = torch.randn(100, 100, dtype=torch.complex64, device=device)
            z_complex = x_complex * y_complex
            print("✓ Complex number operations successful")

        except Exception as e:
            print(f"✗ ROCm math operation test failed - {type(e).__name__}: {e}")

    else:
        print("✗ CUDA not available - ROCm setup may be incomplete")


def test_onnxruntime_setup():
    """Test ONNX Runtime setup with ROCm."""
    print_section("2. ONNX Runtime Setup")

    try:
        import onnxruntime as ort

        print(f"ONNX Runtime version: {ort.__version__}")

        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available execution providers: {providers}")

        # Test ROCm provider
        if "ROCMExecutionProvider" in providers:
            print("✓ ROCMExecutionProvider is available")

            # Test creating a session with ROCm provider
            try:
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 2  # Warning level
                session = ort.InferenceSession(
                    "dummy_model.onnx",
                    sess_options=session_options,
                    providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                )
                print("✓ Successfully created session with ROCMExecutionProvider")

                # Test ROCm provider capabilities
                try:
                    # Create a simple model for testing
                    import numpy as np

                    # Define a simple model that does matrix multiplication
                    from onnx import TensorProto, helper, numpy_helper

                    # Define the graph
                    node1 = helper.make_node(
                        "MatMul",
                        inputs=["input1", "input2"],
                        outputs=["output"],
                    )

                    # Create the graph
                    graph = helper.make_graph(
                        [node1],
                        "test_graph",
                        [
                            helper.make_tensor_value_info(
                                "input1", TensorProto.FLOAT, [2, 2]
                            ),
                            helper.make_tensor_value_info(
                                "input2", TensorProto.FLOAT, [2, 2]
                            ),
                        ],
                        [
                            helper.make_tensor_value_info(
                                "output", TensorProto.FLOAT, [2, 2]
                            )
                        ],
                    )

                    # Create the model
                    model = helper.make_model(graph)

                    # Test inference
                    session = ort.InferenceSession(
                        model.SerializeToString(),
                        providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                    )

                    # Run inference
                    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
                    input2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
                    result = session.run(None, {"input1": input1, "input2": input2})

                    expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
                    if np.allclose(result[0], expected):
                        print("✓ ROCm provider inference test passed")
                    else:
                        print("✗ ROCm provider inference test failed")
                        print(f"  Expected: {expected}")
                        print(f"  Got: {result[0]}")

                except Exception as e:
                    print(f"✗ ROCm inference test failed - {type(e).__name__}: {e}")

            except Exception as e:
                print(
                    f"✗ Failed to create session with ROCMExecutionProvider: {type(e).__name__}: {e}"
                )
        else:
            print("✗ ROCMExecutionProvider not available")

        # Test CUDA provider as fallback
        if "CUDAExecutionProvider" in providers:
            print(
                "✓ CUDAExecutionProvider is available (can be used as fallback for AMD)"
            )

            # Test creating a session with CUDA provider
            try:
                session_options = ort.SessionOptions()
                session = ort.InferenceSession(
                    "dummy_model.onnx",
                    sess_options=session_options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                print("✓ Successfully created session with CUDAExecutionProvider")

                # Test CUDA provider capabilities
                try:
                    # Reuse the same model test as above
                    import numpy as np
                    from onnx import TensorProto, helper

                    # Create the graph
                    node1 = helper.make_node(
                        "MatMul",
                        inputs=["input1", "input2"],
                        outputs=["output"],
                    )

                    graph = helper.make_graph(
                        [node1],
                        "test_graph",
                        [
                            helper.make_tensor_value_info(
                                "input1", TensorProto.FLOAT, [2, 2]
                            ),
                            helper.make_tensor_value_info(
                                "input2", TensorProto.FLOAT, [2, 2]
                            ),
                        ],
                        [
                            helper.make_tensor_value_info(
                                "output", TensorProto.FLOAT, [2, 2]
                            )
                        ],
                    )

                    model = helper.make_model(graph)

                    # Test inference
                    session = ort.InferenceSession(
                        model.SerializeToString(),
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )

                    # Run inference
                    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
                    input2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
                    result = session.run(None, {"input1": input1, "input2": input2})

                    expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
                    if np.allclose(result[0], expected):
                        print("✓ CUDA provider inference test passed")
                    else:
                        print("✗ CUDA provider inference test failed")
                        print(f"  Expected: {expected}")
                        print(f"  Got: {result[0]}")

                except Exception as e:
                    print(f"✗ CUDA inference test failed - {type(e).__name__}: {e}")

            except Exception as e:
                print(
                    f"✗ Failed to create session with CUDAExecutionProvider: {type(e).__name__}: {e}"
                )
        else:
            print("✗ CUDAExecutionProvider not available")

        # Additional ROCm-specific checks
        try:
            # Check for ROCm-specific environment variables
            rocm_vars = [v for v in os.environ.keys() if "rocm" in v.lower()]
            if rocm_vars:
                print("ROCm environment variables detected:")
                for var in rocm_vars:
                    print(f"  {var}: {os.environ[var]}")

            # Check for ROCm installation paths
            rocm_paths = ["/opt/rocm", "/usr/lib/rocm"]
            for path in rocm_paths:
                if os.path.exists(path):
                    print(f"✓ ROCm installation found at {path}")
                    # List contents to verify
                    try:
                        contents = os.listdir(path)
                        if "lib" in contents or "bin" in contents:
                            print(f"  ROCm components detected in {path}")
                    except:
                        pass

        except Exception as e:
            print(f"✗ ROCm environment check failed - {type(e).__name__}: {e}")

    except ImportError:
        print("✗ ONNX Runtime not installed")
    except Exception as e:
        print(f"✗ Error testing ONNX Runtime: {type(e).__name__}: {e}")


def test_stft_operations():
    """Test STFT operations which are critical for audio processing."""
    print_section("3. STFT Operations")

    if not torch.cuda.is_available():
        print("✗ CUDA not available - skipping STFT tests")
        return

    try:
        # Test window creation
        print("Testing Hann window creation...")
        window = torch.hann_window(2048, periodic=True)
        window_gpu = window.to("cuda")
        print(
            f"✓ Hann window created and moved to GPU: shape={window_gpu.shape}, device={window_gpu.device}"
        )

        # Test different window types
        try:
            # Test Hamming window
            hamming_window = torch.hamming_window(2048, periodic=True).to("cuda")
            print("✓ Hamming window created on GPU")

            # Test Blackman window
            blackman_window = torch.blackman_window(2048, periodic=True).to("cuda")
            print("✓ Blackman window created on GPU")

            # Test custom window
            custom_window = torch.ones(2048, device="cuda")
            print("✓ Custom window created on GPU")

        except Exception as e:
            print(f"✗ Window type test failed - {type(e).__name__}: {e}")

        # Test STFT on GPU
        x_gpu = torch.randn(2, 44100).to("cuda")

        # Try different n_fft values
        for n_fft in [1024, 2048]:
            hop_length = n_fft // 4
            try:
                result_gpu = torch.stft(
                    x_gpu,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window_gpu[:n_fft].to("cuda"),
                    center=True,
                    return_complex=True,
                )
                print(
                    f"✓ STFT with n_fft={n_fft}: shape={result_gpu.shape}, device={result_gpu.device}"
                )
            except Exception as e:
                print(f"✗ STFT with n_fft={n_fft}: FAILED - {type(e).__name__}: {e}")

        # Test memory layout of result
        if result_gpu.is_contiguous():
            print(f"  Result is contiguous")
        else:
            print(f"  Result is not contiguous")

        # Test complex tensor operations
        print("\nTesting complex tensor operations...")
        try:
            x_complex = torch.randn(2, 1025, 100, 2, device="cuda")
            x_complex = torch.view_as_complex(x_complex)
            print(f"✓ Complex tensor created: {x_complex.shape}, {x_complex.dtype}")

            # Test different complex operations
            try:
                # Test complex multiplication
                y_complex = torch.randn(
                    2, 1025, 100, dtype=torch.complex64, device="cuda"
                )
                z_complex = x_complex * y_complex
                print("✓ Complex multiplication successful")

                # Test complex addition
                w_complex = x_complex + y_complex
                print("✓ Complex addition successful")

                # Test complex magnitude
                mag = torch.abs(x_complex)
                print("✓ Complex magnitude calculation successful")

                # Test complex angle
                angle = torch.angle(x_complex)
                print("✓ Complex angle calculation successful")

            except Exception as e:
                print(f"✗ Complex operation test failed - {type(e).__name__}: {e}")

            # Test istft
            result_istft = torch.istft(
                x_complex,
                n_fft=2048,
                hop_length=512,
                window=window_gpu,
                center=True,
                length=44100,
            )
            print(f"✓ ISTFT completed: {result_istft.shape}, {result_istft.device}")

            # Test different istft parameters
            try:
                # Test with different length
                result_istft_long = torch.istft(
                    x_complex,
                    n_fft=2048,
                    hop_length=512,
                    window=window_gpu,
                    center=True,
                    length=48000,
                )
                print(
                    f"✓ ISTFT with longer length completed: {result_istft_long.shape}"
                )

            except Exception as e:
                print(f"✗ ISTFT parameter test failed - {type(e).__name__}: {e}")

        except Exception as e:
            print(f"✗ Complex tensor operations: FAILED - {type(e).__name__}: {e}")

        # Test ROCm-specific STFT optimizations
        try:
            print("\nTesting ROCm-specific STFT optimizations...")

            # Test with different data types
            x_float64 = torch.randn(2, 44100, dtype=torch.float64, device="cuda")
            result_float64 = torch.stft(
                x_float64,
                n_fft=2048,
                hop_length=512,
                window=window_gpu[:2048].to(torch.float64),
                center=True,
                return_complex=True,
            )
            print("✓ STFT with float64 precision successful")

            # Test with half precision
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).major >= 5
            ):
                x_float16 = torch.randn(2, 44100, dtype=torch.float16, device="cuda")
                window_float16 = window_gpu[:2048].to(torch.float16)
                result_float16 = torch.stft(
                    x_float16,
                    n_fft=2048,
                    hop_length=512,
                    window=window_float16,
                    center=True,
                    return_complex=True,
                )
                print("✓ STFT with float16 precision successful")

        except Exception as e:
            print(f"✗ ROCm STFT optimization test failed - {type(e).__name__}: {e}")

    except Exception as e:
        print(f"✗ STFT operations: FAILED - {type(e).__name__}: {e}")


def test_memory_allocation():
    """Test GPU memory allocation patterns and identify potential memory issues."""
    print_section("4. GPU Memory Allocation")

    if not torch.cuda.is_available():
        print("✗ CUDA not available - skipping memory tests")
        return

    try:
        # Get detailed GPU memory information
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(
            f"GPU memory max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
        )
        print(
            f"GPU memory max reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB"
        )

        # Get GPU properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"GPU Name: {props.name}")
        print(f"GPU Compute Capability: {props.major}.{props.minor}")
        print(f"GPU Total Memory: {props.total_memory / 1024**2:.2f} MB")
        print(f"GPU Multi processor count: {props.multi_processor_count}")

        # Test different tensor sizes with memory monitoring
        sizes = [1024, 2048, 4096, 8192, 16384]
        for size in sizes:
            try:
                # Report memory before allocation
                mem_before = torch.cuda.memory_allocated() / 1024**2

                # Allocate a large tensor
                tensor = torch.randn(size, size, device="cuda")

                # Report memory after allocation
                mem_after = torch.cuda.memory_allocated() / 1024**2

                print(
                    f"✓ Successfully allocated {size}x{size} tensor ({tensor.element_size() * tensor.nelement() / 1024**2:.2f} MB)"
                    f", memory change: +{mem_after - mem_before:.2f} MB"
                )

                # Perform an operation
                result = torch.mm(tensor, tensor)
                print(f"✓ Matrix multiplication completed")

                # Clean up
                del tensor, result
                torch.cuda.empty_cache()

                # Report memory after cleanup
                mem_cleanup = torch.cuda.memory_allocated() / 1024**2
                print(f"  Memory after cleanup: {mem_cleanup:.2f} MB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"✗ Allocation of {size}x{size} tensor: OOM - {e}")
                    break
                else:
                    print(f"✗ Allocation of {size}x{size} tensor: FAILED - {e}")
                    break
            except Exception as e:
                print(
                    f"✗ Allocation of {size}x{size} tensor: FAILED - {type(e).__name__}: {e}"
                )
                break

        # Test memory fragmentation by allocating and deallocating in random order
        try:
            print("\nTesting memory fragmentation...")
            tensors = []
            sizes_mb = [16, 32, 64, 128, 256, 512]

            for i in range(20):  # Create 20 random allocations
                size_mb = random.choice(sizes_mb)
                size = int(
                    (size_mb * 1024**2) / 4
                )  # Convert MB to number of floats (4 bytes each)
                size = int(size**0.5)  # Make it a square tensor

                try:
                    tensor = torch.randn(size, size, device="cuda")
                    tensors.append(tensor)

                    if len(tensors) % 5 == 0:  # Deallocate every 5th allocation
                        del tensors[0]
                        tensors = tensors[1:]
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(
                            f"✗ Memory fragmentation test failed at iteration {i}: OOM"
                        )
                        break

            else:
                print("✓ Memory fragmentation test completed successfully")

            # Clean up remaining tensors
            for t in tensors:
                del t
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ Memory fragmentation test failed - {type(e).__name__}: {e}")

        # Test large contiguous allocation
        try:
            print("\nTesting large contiguous allocation...")
            # Try to allocate 80% of free memory as a single tensor
            free_memory = props.total_memory - torch.cuda.memory_reserved()
            target_memory = int(free_memory * 0.8)
            elements = target_memory // 4  # 4 bytes per float
            side_length = int(elements**0.5)

            print(
                f"Attempting to allocate {target_memory / 1024**2:.2f} MB ({side_length}x{side_length} tensor)"
            )

            tensor = torch.randn(side_length, side_length, device="cuda")
            print(f"✓ Successfully allocated large contiguous tensor")

            # Test operation on large tensor
            result = torch.mm(tensor, tensor)
            print(f"✓ Matrix multiplication completed on large tensor")

            del tensor, result
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ Large contiguous allocation failed: OOM")
            else:
                print(f"✗ Large contiguous allocation failed - {type(e).__name__}: {e}")
        except Exception as e:
            print(
                f"✗ Large contiguous allocation test failed - {type(e).__name__}: {e}"
            )

        # Test ROCm-specific memory operations
        print("\nTesting ROCm-specific memory operations...")
        try:
            # Test pinned memory (important for ROCm)
            pinned_tensor = torch.randn(1024, 1024, pin_memory=True)
            print("✓ Pinned memory allocation successful")
            del pinned_tensor

            # Test non-blocking operations
            src_tensor = torch.randn(1024, 1024, device="cuda")
            dst_tensor = torch.randn(1024, 1024, device="cuda")
            dst_tensor.copy_(src_tensor, non_blocking=True)
            print("✓ Non-blocking copy successful")

            # Test memory pinning with CPU tensor
            cpu_tensor = torch.randn(1024, 1024)
            cpu_tensor = cpu_tensor.pin_memory()
            gpu_tensor = cpu_tensor.to("cuda", non_blocking=True)
            print("✓ Pinned memory transfer successful")

        except Exception as e:
            print(f"✗ ROCm memory operation failed - {type(e).__name__}: {e}")

    except Exception as e:
        print(f"✗ Memory allocation tests: FAILED - {type(e).__name__}: {e}")


def test_model_types():
    """Test loading and running all model types on ROCm."""
    print_section("5. Model Type Testing")

    if not torch.cuda.is_available():
        print("✗ CUDA not available - skipping model tests")
        return

    try:
        # Test importing all model modules
        from audio_separator.separator.roformer.roformer_loader import RoformerLoader
        from audio_separator.separator.uvr_lib_v5.demucs.hdemucs import HDemucs
        from audio_separator.separator.uvr_lib_v5.demucs.pretrained import (
            get_model as get_demucs_model,
        )
        from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer
        from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import (
            MelBandRoformer,
        )
        from audio_separator.separator.uvr_lib_v5.tfc_tdf_v3 import TFC_TDF_net

        print("✓ Successfully imported all model modules")

        # Test Roformer models
        print_section("5.1 Roformer Models")

        # Test BSRoformer
        try:
            # Use proper freqs_per_bands that sums to 1025 for n_fft=2048
            model = BSRoformer(
                dim=512,
                depth=2,
                freqs_per_bands=(
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    48,
                    48,
                    48,
                    48,
                    48,
                    48,
                    48,
                    48,
                    128,
                    127,
                ),
                stft_n_fft=2048,
                stft_hop_length=512,
                stft_win_length=2048,
            )
            model.to("cuda")
            model.eval()

            print("✓ Successfully created and moved BSRoformer model to GPU")

            # Test with a small input
            test_input = torch.randn(1, 44100, device="cuda", requires_grad=True)
            output = model(test_input)

            print(f"✓ Model forward pass completed: output shape {output.shape}")

            # Test gradient computation
            try:
                model.train()
                loss = torch.nn.functional.l1_loss(output, torch.zeros_like(output))
                loss.backward()
                print("✓ Gradient computation successful")
                model.eval()
            except Exception as e:
                print(f"✗ Gradient computation failed - {type(e).__name__}: {e}")

        except Exception as e:
            print(f"✗ BSRoformer model operations: FAILED - {type(e).__name__}: {e}")

        # Test MelBandRoformer
        try:
            model = MelBandRoformer(
                dim=512,
                depth=2,
                num_bands=60,
                stft_n_fft=2048,
                stft_hop_length=512,
                stft_win_length=2048,
            )
            model.to("cuda")
            model.eval()

            print("✓ Successfully created and moved MelBandRoformer model to GPU")

            # Test with a small input
            test_input = torch.randn(1, 44100, device="cuda", requires_grad=True)
            output = model(test_input)

            print(f"✓ Model forward pass completed: output shape {output.shape}")

            # Test gradient computation
            try:
                model.train()
                loss = torch.nn.functional.l1_loss(output, torch.zeros_like(output))
                loss.backward()
                print("✓ Gradient computation successful")
                model.eval()
            except Exception as e:
                print(f"✗ Gradient computation failed - {type(e).__name__}: {e}")

        except Exception as e:
            print(
                f"✗ MelBandRoformer model operations: FAILED - {type(e).__name__}: {e}"
            )

        # Test RoformerLoader
        try:
            loader = RoformerLoader()
            print("✓ Successfully created RoformerLoader")

            # Test configuration validation
            config = {
                "dim": 512,
                "depth": 2,
                "freqs_per_bands": (
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    48,
                    48,
                    48,
                    48,
                    48,
                    48,
                    48,
                    48,
                    128,
                    127,
                ),
                "stft_n_fft": 2048,
                "stft_hop_length": 512,
                "stft_win_length": 2048,
            }

            is_valid = loader.validate_configuration(config, "bs_roformer")
            if is_valid:
                print("✓ Configuration validation passed")
            else:
                print("✗ Configuration validation failed")

        except Exception as e:
            print(f"✗ RoformerLoader test failed - {type(e).__name__}: {e}")

        # Test Demucs models
        print_section("5.2 Demucs Models")
        try:
            # Test HDemucs model architecture
            demucs_model = HDemucs(sources=["drums", "bass", "other", "vocals"])
            demucs_model.to("cuda")
            demucs_model.train()  # Keep in training mode for backward pass

            print("✓ Successfully created and moved Demucs model to GPU")

            # Test with a small input
            test_input = torch.randn(1, 2, 44100, device="cuda", requires_grad=True)
            output = demucs_model(test_input)

            print(f"✓ Demucs model forward pass completed: output shape {output.shape}")

            # Test gradient computation
            try:
                loss = torch.nn.functional.l1_loss(output, torch.zeros_like(output))
                loss.backward()
                print("✓ Demucs gradient computation successful")
                demucs_model.eval()
            except Exception as e:
                print(f"✗ Demucs gradient computation failed - {type(e).__name__}: {e}")

        except Exception as e:
            print(f"✗ Demucs model operations: FAILED - {type(e).__name__}: {e}")

        # Test MDX models
        print_section("5.3 MDX Models")
        try:
            # Test TFC_TDF_net model
            from ml_collections import ConfigDict

            # Create a complete config for testing
            test_config = ConfigDict(
                {
                    "audio": {
                        "n_fft": 2048,
                        "hop_length": 512,
                        "dim_f": 1025,
                        "num_channels": 2,
                    },
                    "inference": {
                        "dim_t": 256,
                    },
                    "training": {
                        "instruments": ["vocals", "instrumental"],
                        "target_instrument": "vocals",
                    },
                    "model": {
                        "norm": "Identity",
                        "act": "GELU",
                        "num_subbands": 1,
                        "num_scales": 2,
                        "scale": [2, 2],
                        "num_blocks_per_scale": 2,
                        "num_channels": 16,
                        "growth": 8,
                        "bottleneck_factor": 4,
                    },
                }
            )

            print("Creating TFC_TDF_net model...")
            try:
                model = TFC_TDF_net(test_config, device="cuda")
                print(f"Model created. Model device: {next(model.parameters()).device}")

                # Print first few parameters to verify model creation
                for i, (name, param) in enumerate(model.named_parameters()):
                    if i < 3:  # Just show first few
                        print(f"  {name}: {param.shape} {param.device}")
                    else:
                        print("  ...")
                        break

            except Exception as e:
                print(f"✗ Failed to create model - {type(e).__name__}: {e}")
                return

            print("Moving model to CUDA...")
            try:
                model.to("cuda")
                print(
                    f"Model moved to CUDA. Current device: {next(model.parameters()).device}"
                )

                # Verify parameters are on CUDA
                for i, (name, param) in enumerate(model.named_parameters()):
                    if i < 3:  # Just show first few
                        print(f"  {name}: {param.shape} {param.device}")
                    else:
                        print("  ...")
                        break

            except Exception as e:
                print(f"✗ Failed to move model to CUDA - {type(e).__name__}: {e}")
                return

            model.eval()
            print("✓ Successfully created and moved TFC_TDF_net model to GPU")

            # Test with a small input
            test_input = torch.randn(1, 2, 44100, device="cuda", requires_grad=True)
            print(f"Test input created on device: {test_input.device}")

            try:
                print("Running forward pass...")
                print(f"Input shape: {test_input.shape}")
                output = model(test_input)
                print(
                    f"✓ MDX model forward pass completed: output shape {output.shape}"
                )

                # Test gradient computation
                try:
                    model.train()
                    print("Computing loss...")
                    loss = torch.nn.functional.l1_loss(output, torch.zeros_like(output))
                    print(f"Loss computed: {loss.item()}")
                    print("Running backward pass...")
                    loss.backward()
                    print("✓ MDX gradient computation successful")
                    model.eval()
                except Exception as e:
                    print(
                        f"✗ MDX gradient computation failed - {type(e).__name__}: {e}"
                    )
            except Exception as e:
                print(f"✗ MDX forward pass failed - {type(e).__name__}: {e}")

        except Exception as e:
            print(f"✗ MDX model operations: FAILED - {type(e).__name__}: {e} {str(e)}")

    except ImportError as e:
        print(f"✗ Failed to import model modules: {e}")
    except Exception as e:
        print(f"✗ Model loading: FAILED - {type(e).__name__}: {e}")


def main():
    """Main function to run all tests."""
    print("ROCm Debug Script for audio-separator")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # Print system information
    print_section("0. System Information")
    system_info = get_system_info()
    if "cpu_model" in system_info:
        print(f"CPU: {system_info['cpu_model']}")
    if "cpu_cores" in system_info:
        print(f"CPU Cores: {system_info['cpu_cores']}")
    if "memory_total" in system_info:
        print(f"Memory: {system_info['memory_total']:.0f} MB")
    if "kernel_version" in system_info:
        print(f"Kernel: {system_info['kernel_version']}")

    # Print ROCm information
    rocm_info = get_rocm_info()
    if "version" in rocm_info:
        print(f"ROCm Version: {rocm_info['version']}")
    if "hip_version" in rocm_info:
        print(f"HIP Version: {rocm_info['hip_version']}")

    # Print environment variables
    print(
        f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set')}"
    )
    print(f"PYTORCH_ROCM_ARCH: {os.environ.get('PYTORCH_ROCM_ARCH', 'Not set')}")

    # Check for relevant environment variables
    relevant_vars = [
        v
        for v in os.environ.keys()
        if "rocm" in v.lower() or "gpu" in v.lower() or "hip" in v.lower()
    ]
    if relevant_vars:
        print("Relevant environment variables:")
        for var in relevant_vars:
            print(f"  {var}: {os.environ[var]}")

    # Check PyTorch ROCm build
    print_section("0.1 PyTorch ROCm Build Check")
    if "+rocm" in torch.__version__:
        print("✓ PyTorch built with ROCm support")
    else:
        print("✗ PyTorch not built with ROCm support - consider reinstalling with ROCm")

    # Check for ROCm-specific environment variables
    print_section("0.2 ROCm Environment Variables")
    rocm_vars = [v for v in os.environ.keys() if "rocm" in v.lower()]
    if rocm_vars:
        for var in rocm_vars:
            print(f"  {var}: {os.environ[var]}")
    else:
        print("  No ROCm-specific environment variables set")

    # Check for known ROCm issues
    print_section("0.3 Known ROCm Issues Check")
    if torch.cuda.is_available():
        # Check for gfx1032 issue
        if "gfx1032" in str(torch.cuda.get_device_properties(0)):
            if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
                print("✗ gfx1032 GPU detected but HSA_OVERRIDE_GFX_VERSION not set")
                print("  Recommendation: export HSA_OVERRIDE_GFX_VERSION=10.3.2")
            else:
                print("✓ gfx1032 GPU workaround enabled")

        # Check for autocast issues
        if hasattr(torch.cuda, "amp"):
            print("✓ CUDA AMP (autocast) available")
        else:
            print("✗ CUDA AMP (autocast) not available - may affect performance")

    # Check memory layout
    print_section("0.4 Memory Layout Check")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Test different memory layouts
        test_tensor = torch.randn(1024, 1024, device=device)
        contiguous_tensor = test_tensor.contiguous()
        if contiguous_tensor.is_contiguous():
            print("✓ Contiguous memory allocation working")
        else:
            print("✗ Contiguous memory allocation failed")

        # Test non-contiguous tensor
        non_contiguous = test_tensor.t()
        if not non_contiguous.is_contiguous():
            print("✓ Non-contiguous memory layout working")
        else:
            print("✗ Non-contiguous memory layout issue")

    # Run only MDX model test for focused debugging
    test_model_types()

    print("\n" + "=" * 50)
    print("Debugging complete. Review the results above to identify issues.")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
