"""
Data models and enums for the Greeks calculator
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class OptionType(Enum):
    """Option type enumeration"""
    CALL = 'CE'
    PUT = 'PE'
    CE = 'CE'
    PE = 'PE'

    @classmethod
    def from_string(cls, value: str):
        """Convert string to OptionType"""
        value = value.upper()
        if value in ['C', 'CALL', 'CE']:
            return cls.CALL
        elif value in ['P', 'PUT', 'PE']:
            return cls.PUT
        else:
            raise ValueError(f"Invalid option type: {value}")


class AssetClass(Enum):
    """Asset class enumeration"""
    EQUITY = 'EQUITY'
    CRYPTO = 'CRYPTO'


class VolatilityModel(Enum):
    """Supported volatility models"""
    BLACK_SCHOLES = 'BLACK_SCHOLES'
    SABR = 'SABR'
    SVI = 'SVI'


@dataclass
class GreeksResult:
    """
    Result object for Greeks calculations

    Attributes:
        delta: Option delta (-1 to 1)
        gamma: Option gamma (always positive)
        theta: Option theta (negative for long positions)
        vega: Option vega
        rho: Option rho (interest rate sensitivity)

        calculation_method: Method used for calculation
        implied_volatility: IV used in calculation
        hours_to_expiry: Hours remaining to expiry

        is_valid: Whether calculation succeeded
        errors: List of error messages if any
        warnings: List of warning messages if any
    """
    # Core Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float = 0.0

    # Metadata
    calculation_method: str = ""
    implied_volatility: float = 0.0
    hours_to_expiry: float = 0.0

    # Error handling
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def create_invalid(cls, error_message: str, partial_results: dict = None):
        """Create an invalid result with safe defaults"""
        result = cls(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            is_valid=False,
            errors=[error_message]
        )

        # If we have partial results, update them
        if partial_results:
            for key, value in partial_results.items():
                if hasattr(result, key):
                    setattr(result, key, value)

        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for easy use"""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'implied_volatility': self.implied_volatility,
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings
        }

    def __repr__(self):
        if self.is_valid:
            return (f"GreeksResult(δ={self.delta:.4f}, γ={self.gamma:.4f}, "
                    f"θ={self.theta:.2f}, ν={self.vega:.2f})")
        else:
            return f"GreeksResult(invalid, errors={self.errors})"


@dataclass
class MarketData:
    """Market data container"""
    spot: float
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0

    def validate(self):
        """Validate market data"""
        if self.spot <= 0:
            raise ValueError("Spot price must be positive")
        if self.risk_free_rate < 0:
            raise ValueError("Risk-free rate cannot be negative")
        if self.dividend_yield < 0:
            raise ValueError("Dividend yield cannot be negative")


@dataclass
class PositionGreeks:
    """Greeks for a single position including position size"""
    token: int
    strike: float
    option_type: str
    quantity: float
    multiplier: float

    # Raw Greeks (per contract)
    unit_delta: float
    unit_gamma: float
    unit_theta: float
    unit_vega: float

    # Position Greeks (weighted by quantity * multiplier)
    position_delta: float
    position_gamma: float
    position_theta: float
    position_vega: float

    # Additional info
    market_price: float
    implied_volatility: float
    hours_to_expiry: float


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio-level Greeks"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float

    # Risk metrics
    total_value: float
    net_premium: float

    # Position breakdown
    positions: List[PositionGreeks]

    # Metadata
    calculation_time: str
    underlying: str
    spot_price: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'total_delta': self.total_delta,
            'total_gamma': self.total_gamma,
            'total_theta': self.total_theta,
            'total_vega': self.total_vega,
            'total_value': self.total_value,
            'net_premium': self.net_premium,
            'position_count': len(self.positions),
            'calculation_time': self.calculation_time,
            'underlying': self.underlying,
            'spot_price': self.spot_price
        }