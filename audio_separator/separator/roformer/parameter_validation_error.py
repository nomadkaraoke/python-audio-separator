"""
Parameter validation error exception.
Raised when model parameters are invalid or incompatible.
"""

from typing import Any, Optional


class ParameterValidationError(Exception):
    """
    Exception raised when model parameters are invalid.
    
    This exception provides detailed information about what parameter
    was invalid, what was expected, and suggestions for fixing the issue.
    """
    
    def __init__(
        self, 
        parameter_name: str, 
        expected_type: str, 
        actual_value: Any, 
        suggested_fix: str,
        context: Optional[str] = None
    ):
        """
        Initialize parameter validation error.
        
        Args:
            parameter_name: Name of the invalid parameter
            expected_type: Expected type or description of valid values
            actual_value: The actual value that was provided
            suggested_fix: Suggestion for how to fix the issue
            context: Additional context about where the error occurred
        """
        self.parameter_name = parameter_name
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.suggested_fix = suggested_fix
        self.context = context
        
        # Create detailed error message
        message = self._create_error_message()
        super().__init__(message)
    
    def _create_error_message(self) -> str:
        """Create a detailed error message."""
        actual_type = type(self.actual_value).__name__
        
        message_parts = [
            f"Invalid parameter '{self.parameter_name}': ",
            f"expected {self.expected_type}, got {actual_type} ({self.actual_value})"
        ]
        
        if self.context:
            message_parts.append(f" in {self.context}")
        
        message_parts.append(f". Suggestion: {self.suggested_fix}")
        
        return "".join(message_parts)
    
    def get_error_details(self) -> dict:
        """Get error details as a dictionary."""
        return {
            'parameter_name': self.parameter_name,
            'expected_type': self.expected_type,
            'actual_value': self.actual_value,
            'actual_type': type(self.actual_value).__name__,
            'suggested_fix': self.suggested_fix,
            'context': self.context,
            'error_message': str(self)
        }
    
    @classmethod
    def missing_parameter(cls, parameter_name: str, parameter_type: str, context: Optional[str] = None) -> 'ParameterValidationError':
        """
        Create error for missing required parameter.
        
        Args:
            parameter_name: Name of missing parameter
            parameter_type: Expected type of the parameter
            context: Context where parameter is missing
            
        Returns:
            ParameterValidationError for missing parameter
        """
        return cls(
            parameter_name=parameter_name,
            expected_type=parameter_type,
            actual_value=None,
            suggested_fix=f"Add '{parameter_name}' parameter with {parameter_type} value",
            context=context
        )
    
    @classmethod
    def wrong_type(cls, parameter_name: str, expected_type: str, actual_value: Any, context: Optional[str] = None) -> 'ParameterValidationError':
        """
        Create error for wrong parameter type.
        
        Args:
            parameter_name: Name of parameter with wrong type
            expected_type: Expected type
            actual_value: Actual value provided
            context: Context where error occurred
            
        Returns:
            ParameterValidationError for type mismatch
        """
        return cls(
            parameter_name=parameter_name,
            expected_type=expected_type,
            actual_value=actual_value,
            suggested_fix=f"Change '{parameter_name}' to {expected_type}",
            context=context
        )
    
    @classmethod
    def out_of_range(cls, parameter_name: str, valid_range: str, actual_value: Any, context: Optional[str] = None) -> 'ParameterValidationError':
        """
        Create error for parameter value out of valid range.
        
        Args:
            parameter_name: Name of parameter out of range
            valid_range: Description of valid range
            actual_value: Actual value provided
            context: Context where error occurred
            
        Returns:
            ParameterValidationError for out of range value
        """
        return cls(
            parameter_name=parameter_name,
            expected_type=f"value in range {valid_range}",
            actual_value=actual_value,
            suggested_fix=f"Set '{parameter_name}' to a value within {valid_range}",
            context=context
        )
    
    @classmethod
    def incompatible_parameters(cls, parameter_names: list, issue_description: str, suggested_fix: str, context: Optional[str] = None) -> 'ParameterValidationError':
        """
        Create error for incompatible parameter combination.
        
        Args:
            parameter_names: List of parameter names that are incompatible
            issue_description: Description of the incompatibility
            suggested_fix: How to fix the incompatibility
            context: Context where error occurred
            
        Returns:
            ParameterValidationError for incompatible parameters
        """
        parameter_list = ", ".join(parameter_names)
        return cls(
            parameter_name=parameter_list,
            expected_type="compatible parameter combination",
            actual_value=issue_description,
            suggested_fix=suggested_fix,
            context=context
        )
    
    @classmethod
    def invalid_normalization(cls, norm_value: Any, supported_norms: list, context: Optional[str] = None) -> 'ParameterValidationError':
        """
        Create error for invalid normalization configuration.
        
        Args:
            norm_value: The invalid normalization value
            supported_norms: List of supported normalization types
            context: Context where error occurred
            
        Returns:
            ParameterValidationError for invalid normalization
        """
        supported_list = ", ".join(f"'{norm}'" for norm in supported_norms)
        return cls(
            parameter_name="norm",
            expected_type=f"one of: {supported_list}",
            actual_value=norm_value,
            suggested_fix=f"Use one of the supported normalization types: {supported_list}",
            context=context
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ParameterValidationError(parameter='{self.parameter_name}', expected='{self.expected_type}', actual={self.actual_value})"
