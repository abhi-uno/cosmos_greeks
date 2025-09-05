"""
Custom exceptions for the Greeks calculator
"""


class GreeksCalculationError(Exception):
    """Base exception for Greeks calculation errors"""
    pass


class ValidationError(GreeksCalculationError):
    """Raised when input validation fails"""
    pass


class VolatilityCalculationError(GreeksCalculationError):
    """Raised when implied volatility calculation fails"""
    pass


class QuantLibError(GreeksCalculationError):
    """Raised when QuantLib operations fail"""
    pass


class TimeToExpiryError(GreeksCalculationError):
    """Raised when time to expiry calculation has issues"""
    pass