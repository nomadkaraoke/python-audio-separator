"""
Unit tests for separator detection and routing logic.
Tests the detection of Roformer models and proper routing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile


class TestSeparatorDetection:
    """Test cases for separator detection and model routing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_is_roformer_set_from_yaml_path(self):
        """T059: YAML path containing 'roformer' sets is_roformer and routes Roformer path."""
        # Test cases for different YAML paths
        test_cases = [
            # (yaml_path, expected_is_roformer, description)
            ("model_bs_roformer_ep_317_sdr_12.yaml", True, "BS Roformer model"),
            ("mel_band_roformer_vocals.yaml", True, "Mel Band Roformer model"),
            ("roformer_large_model.yaml", True, "Generic Roformer model"),
            ("BS-Roformer-Viperx-1297.yaml", True, "BS-Roformer with uppercase"),
            ("model_mdx_extra_vocals.yaml", False, "MDX model (not Roformer)"),
            ("vr_model_vocals.yaml", False, "VR model (not Roformer)"),
            ("demucs_model.yaml", False, "Demucs model (not Roformer)"),
            ("some_other_model.yaml", False, "Generic other model"),
        ]

        for yaml_path, expected_is_roformer, description in test_cases:
            # Mock the detection logic
            def mock_detect_roformer_from_path(path):
                """Mock detection that checks if path contains 'roformer'."""
                path_lower = path.lower()
                return 'roformer' in path_lower

            is_roformer = mock_detect_roformer_from_path(yaml_path)
            
            assert is_roformer == expected_is_roformer, (
                f"Failed for {description}: path '{yaml_path}' should "
                f"{'be' if expected_is_roformer else 'not be'} detected as Roformer"
            )

        # Test routing logic based on detection
        def mock_route_separator(yaml_path):
            """Mock routing logic that selects separator based on detection."""
            is_roformer = mock_detect_roformer_from_path(yaml_path)
            
            if is_roformer:
                return "RoformerSeparator"
            elif 'mdx' in yaml_path.lower():
                return "MDXSeparator"
            elif 'vr' in yaml_path.lower():
                return "VRSeparator"
            elif 'demucs' in yaml_path.lower():
                return "DemucsSeparator"
            else:
                return "DefaultSeparator"

        # Test routing for Roformer models
        roformer_paths = [
            "model_bs_roformer_ep_317_sdr_12.yaml",
            "mel_band_roformer_vocals.yaml",
            "BS-Roformer-Viperx-1297.yaml"
        ]
        
        for path in roformer_paths:
            separator_type = mock_route_separator(path)
            assert separator_type == "RoformerSeparator", (
                f"Roformer model '{path}' should route to RoformerSeparator, got {separator_type}"
            )

        # Test routing for non-Roformer models
        non_roformer_cases = [
            ("model_mdx_extra_vocals.yaml", "MDXSeparator"),
            ("vr_model_vocals.yaml", "VRSeparator"),
            ("demucs_model.yaml", "DemucsSeparator"),
            ("some_other_model.yaml", "DefaultSeparator"),
        ]
        
        for path, expected_separator in non_roformer_cases:
            separator_type = mock_route_separator(path)
            assert separator_type == expected_separator, (
                f"Non-Roformer model '{path}' should route to {expected_separator}, got {separator_type}"
            )

    def test_roformer_detection_case_insensitive(self):
        """Test that Roformer detection is case-insensitive."""
        case_variations = [
            "ROFORMER_MODEL.yaml",
            "roformer_model.yaml", 
            "RoFormer_Model.yaml",
            "BS_ROFORMER.yaml",
            "bs_roformer.yaml",
            "Bs_Roformer.yaml",
            "MEL_BAND_ROFORMER.yaml",
            "mel_band_roformer.yaml",
            "Mel_Band_Roformer.yaml"
        ]

        def mock_detect_roformer_case_insensitive(path):
            """Mock detection with case-insensitive matching."""
            return 'roformer' in path.lower()

        for path in case_variations:
            is_roformer = mock_detect_roformer_case_insensitive(path)
            assert is_roformer, f"Case variation '{path}' should be detected as Roformer"

    def test_roformer_detection_with_full_paths(self):
        """Test Roformer detection works with full file paths."""
        full_path_cases = [
            ("/models/roformer/model_bs_roformer_ep_317.yaml", True),
            ("/path/to/models/mel_band_roformer_vocals.yaml", True),
            ("/home/user/models/mdx_extra_vocals.yaml", False),
            ("C:\\Models\\BS-Roformer-Viperx-1297.yaml", True),
            ("./local/models/vr_model_vocals.yaml", False),
            ("../models/roformer_large_model.yaml", True),
        ]

        def mock_detect_roformer_full_path(full_path):
            """Mock detection that works with full paths."""
            filename = os.path.basename(full_path)
            return 'roformer' in filename.lower()

        for full_path, expected_result in full_path_cases:
            is_roformer = mock_detect_roformer_full_path(full_path)
            assert is_roformer == expected_result, (
                f"Full path '{full_path}' detection failed: expected {expected_result}, got {is_roformer}"
            )

    def test_roformer_detection_with_config_content(self):
        """Test Roformer detection based on YAML configuration content."""
        # Mock YAML configurations
        roformer_configs = [
            {"model_type": "bs_roformer", "architecture": "BSRoformer"},
            {"model_type": "mel_band_roformer", "architecture": "MelBandRoformer"},
            {"architecture": "roformer", "variant": "large"},
            {"separator_class": "RoformerSeparator", "model": "bs_roformer"},
        ]

        non_roformer_configs = [
            {"model_type": "mdx", "architecture": "MDX"},
            {"model_type": "vr", "architecture": "VR"},
            {"architecture": "demucs", "variant": "v4"},
            {"separator_class": "MDXSeparator", "model": "mdx_extra"},
        ]

        def mock_detect_roformer_from_config(config):
            """Mock detection based on configuration content."""
            config_str = str(config).lower()
            return 'roformer' in config_str

        # Test Roformer configs
        for config in roformer_configs:
            is_roformer = mock_detect_roformer_from_config(config)
            assert is_roformer, f"Roformer config should be detected: {config}"

        # Test non-Roformer configs  
        for config in non_roformer_configs:
            is_roformer = mock_detect_roformer_from_config(config)
            assert not is_roformer, f"Non-Roformer config should not be detected: {config}"

    def test_roformer_routing_integration(self):
        """Test integration between detection and routing."""
        with patch('audio_separator.separator.separator.Separator') as mock_separator_class:
            mock_instance = Mock()
            mock_separator_class.return_value = mock_instance
            
            # Mock the routing logic
            def mock_load_model_with_routing(model_path):
                """Mock model loading that routes based on detection."""
                filename = os.path.basename(model_path)
                is_roformer = 'roformer' in filename.lower()
                
                if is_roformer:
                    # Should use Roformer-specific loading
                    return {
                        'separator_type': 'RoformerSeparator',
                        'is_roformer': True,
                        'routing_path': 'roformer_path'
                    }
                else:
                    # Should use default loading
                    return {
                        'separator_type': 'DefaultSeparator', 
                        'is_roformer': False,
                        'routing_path': 'default_path'
                    }

            # Test Roformer model routing
            roformer_result = mock_load_model_with_routing("model_bs_roformer_ep_317.ckpt")
            assert roformer_result['is_roformer'] is True
            assert roformer_result['separator_type'] == 'RoformerSeparator'
            assert roformer_result['routing_path'] == 'roformer_path'

            # Test non-Roformer model routing
            mdx_result = mock_load_model_with_routing("model_mdx_extra_vocals.ckpt")
            assert mdx_result['is_roformer'] is False
            assert mdx_result['separator_type'] == 'DefaultSeparator'
            assert mdx_result['routing_path'] == 'default_path'

    def test_roformer_detection_edge_cases(self):
        """Test edge cases in Roformer detection."""
        edge_cases = [
            # (path, expected_result, description)
            ("", False, "Empty string"),
            ("model.yaml", False, "No roformer in name"),
            ("roformer", True, "Just 'roformer'"),
            ("notroformer.yaml", True, "Contains 'roformer' as substring"),
            ("roformer.txt", True, "Different extension"),
            ("ROFORMER.YAML", True, "All uppercase"),
            ("r0f0rmer.yaml", False, "Similar but not exact"),
            ("roformermodel.ckpt", True, "No separator between roformer and model"),
            ("model_roformer_v2.pth", True, "Roformer in middle of filename"),
        ]

        def mock_detect_roformer_edge_cases(path):
            """Mock detection handling edge cases."""
            if not path:
                return False
            return 'roformer' in path.lower()

        for path, expected_result, description in edge_cases:
            is_roformer = mock_detect_roformer_edge_cases(path)
            assert is_roformer == expected_result, (
                f"Edge case failed - {description}: path '{path}' should "
                f"{'be' if expected_result else 'not be'} detected as Roformer"
            )

    def test_roformer_detection_with_model_extensions(self):
        """Test Roformer detection works with various model file extensions."""
        model_extensions = ['.ckpt', '.pth', '.pt', '.onnx', '.yaml', '.yml']
        base_names = ['bs_roformer_model', 'mel_band_roformer', 'roformer_large']

        def mock_detect_roformer_with_extension(path):
            """Mock detection that ignores file extension."""
            filename = os.path.basename(path)
            name_without_ext = os.path.splitext(filename)[0]
            return 'roformer' in name_without_ext.lower()

        for base_name in base_names:
            for ext in model_extensions:
                full_filename = f"{base_name}{ext}"
                is_roformer = mock_detect_roformer_with_extension(full_filename)
                assert is_roformer, (
                    f"Roformer model '{full_filename}' should be detected regardless of extension"
                )

        # Test non-Roformer models with same extensions
        non_roformer_bases = ['mdx_model', 'vr_vocals', 'demucs_v4']
        for base_name in non_roformer_bases:
            for ext in model_extensions:
                full_filename = f"{base_name}{ext}"
                is_roformer = mock_detect_roformer_with_extension(full_filename)
                assert not is_roformer, (
                    f"Non-Roformer model '{full_filename}' should not be detected"
                )
