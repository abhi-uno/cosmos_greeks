"""
cosmos_greeks_calculator - Production-ready Greeks calculation library

A unified Greeks calculator supporting 0-DTE to standard options with multiple volatility models.
Designed for both backtesting and live trading environments.
"""

from .calculator import GreeksCalculator, calculate_greeks
from .models import GreeksResult, OptionType, VolatilityModel
from .errors import GreeksCalculationError, ValidationError

__version__ = "1.0.0"
__author__ = "COSMOS Trading System"

__all__ = [
    "GreeksCalculator",
    "calculate_greeks",
    "GreeksResult",
    "OptionType",
    "VolatilityModel",
    "GreeksCalculationError",
    "ValidationError"
]


# Module-level docstring for help()
def get_info():
    """
    Get information about the cosmos_greeks_calculator library

    Example usage:
        import cosmos_greeks_calculator as cgc

        # Simple calculation
        result = cgc.calculate_greeks(
            spot=25000,
            strike=25100,
            expiry_datetime=datetime(2025, 7, 10, 15, 30),
            market_price=45.5,
            option_type='CE'
        )

        # Advanced usage with calculator instance
        calculator = cgc.GreeksCalculator(underlying='NIFTY')
        result = calculator.calculate(...)
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': __doc__.strip(),
        'models': ['BLACK_SCHOLES', 'SABR', 'SVI'],
        'underlyings': ['NIFTY', 'SENSEX', 'BANKNIFTY', 'FINNIFTY']
    }