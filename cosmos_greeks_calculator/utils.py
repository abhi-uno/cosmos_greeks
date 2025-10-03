"""
Utility functions for the Greeks calculator
"""

from datetime import datetime, time, timedelta
import numpy as np
from typing import Union, Optional

from .models import OptionType, AssetClass
from .errors import ValidationError, TimeToExpiryError

# Market constants for equity
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
TRADING_HOURS_PER_DAY = 6.25
TRADING_DAYS_PER_YEAR = 252

# Crypto market constants
CRYPTO_HOURS_PER_YEAR = 8760  # 365 * 24
CRYPTO_EXPIRY_TIME = time(17, 30)  # 5:30 PM IST for Delta Exchange

# Underlying specifications
LOT_SIZES = {
    # Equity indices
    'NIFTY': 50,
    'SENSEX': 10,
    'BANKNIFTY': 15,
    'FINNIFTY': 40,
    # Crypto (Delta Exchange)
    'BTC': 0.001,
    'ETH': 0.01
}

STRIKE_INTERVALS = {
    # Equity indices
    'NIFTY': 50,
    'SENSEX': 100,
    'BANKNIFTY': 100,
    'FINNIFTY': 50,
    # Crypto - None indicates dynamic strikes
    'BTC': None,
    'ETH': None
}


def calculate_hours_to_expiry(expiry_datetime: datetime,
                              current_datetime: Optional[datetime] = None,
                              asset_class: AssetClass = AssetClass.EQUITY) -> float:
    """
    Calculate hours remaining to expiry based on asset class

    Args:
        expiry_datetime: Expiry date and time
        current_datetime: Current date and time (defaults to now)
        asset_class: EQUITY or CRYPTO

    Returns:
        Hours to expiry (trading hours for equity, calendar hours for crypto)

    Raises:
        TimeToExpiryError: If calculation fails
    """
    if asset_class == AssetClass.CRYPTO:
        return calculate_hours_to_expiry_crypto(expiry_datetime, current_datetime)
    else:
        return calculate_hours_to_expiry_equity(expiry_datetime, current_datetime)


def calculate_hours_to_expiry_equity(expiry_datetime: datetime,
                              current_datetime: Optional[datetime] = None) -> float:
    """
    Calculate trading hours remaining to expiry for equity options

    Args:
        expiry_datetime: Expiry date and time
        current_datetime: Current date and time (defaults to now)

    Returns:
        Trading hours to expiry

    Raises:
        TimeToExpiryError: If calculation fails
    """
    try:
        if current_datetime is None:
            current_datetime = datetime.now()

        # If already expired
        if current_datetime >= expiry_datetime:
            return 0.0

        # Same day expiry (0-DTE)
        if current_datetime.date() == expiry_datetime.date():
            # Simple hours calculation for same day
            hours = (expiry_datetime - current_datetime).total_seconds() / 3600

            # But cap at market hours
            if current_datetime.time() < MARKET_OPEN:
                # Before market open
                market_open_today = current_datetime.replace(hour=9, minute=15, second=0)
                hours = (expiry_datetime - market_open_today).total_seconds() / 3600

            return max(0.0, min(hours, TRADING_HOURS_PER_DAY))

        # Multi-day calculation
        trading_hours = 0.0
        current = current_datetime

        # Add remaining hours today
        if current.time() < MARKET_CLOSE:
            if current.time() < MARKET_OPEN:
                # Full day
                trading_hours += TRADING_HOURS_PER_DAY
            else:
                # Partial day
                close_today = current.replace(hour=15, minute=30, second=0)
                trading_hours += (close_today - current).total_seconds() / 3600

        # Move to next day
        current = current.replace(hour=9, minute=15, second=0)
        current += timedelta(days=1)

        # Skip weekends
        while current.weekday() in [5, 6]:  # Saturday, Sunday
            current += timedelta(days=1)

        # Add complete trading days
        while current.date() < expiry_datetime.date():
            trading_hours += TRADING_HOURS_PER_DAY
            current += timedelta(days=1)

            # Skip weekends
            while current.weekday() in [5, 6]:
                current += timedelta(days=1)

        # Add hours on expiry day
        if current.date() == expiry_datetime.date():
            if expiry_datetime.time() <= MARKET_OPEN:
                # Expires before/at market open
                pass
            elif expiry_datetime.time() >= MARKET_CLOSE:
                # Full day
                trading_hours += TRADING_HOURS_PER_DAY
            else:
                # Partial day
                market_open = current.replace(hour=9, minute=15, second=0)
                trading_hours += (expiry_datetime - market_open).total_seconds() / 3600

        return trading_hours

    except Exception as e:
        raise TimeToExpiryError(f"Failed to calculate time to expiry: {str(e)}")


def calculate_hours_to_expiry_crypto(expiry_datetime: datetime,
                                     current_datetime: Optional[datetime] = None) -> float:
    """
    Calculate calendar hours remaining to expiry for crypto options

    Args:
        expiry_datetime: Expiry date and time
        current_datetime: Current date and time (defaults to now)

    Returns:
        Calendar hours to expiry (continuous 24/7 market)

    Raises:
        TimeToExpiryError: If calculation fails
    """
    try:
        if current_datetime is None:
            current_datetime = datetime.now()

        # If already expired
        if current_datetime >= expiry_datetime:
            return 0.0

        # Simple calculation - crypto markets are 24/7
        hours = (expiry_datetime - current_datetime).total_seconds() / 3600

        return max(0.0, hours)

    except Exception as e:
        raise TimeToExpiryError(f"Failed to calculate crypto time to expiry: {str(e)}")


def validate_inputs(spot: float, strike: float, market_price: float,
                    option_type: Union[str, OptionType]) -> OptionType:
    """
    Validate input parameters

    Returns:
        Validated OptionType

    Raises:
        ValidationError: If validation fails
    """
    # Price validations
    if spot <= 0:
        raise ValidationError(f"Spot price must be positive, got {spot}")

    if strike <= 0:
        raise ValidationError(f"Strike price must be positive, got {strike}")

    if market_price < 0:
        raise ValidationError(f"Market price cannot be negative, got {market_price}")

    # Option type validation
    if isinstance(option_type, str):
        try:
            option_type = OptionType.from_string(option_type)
        except ValueError as e:
            raise ValidationError(str(e))
    elif not isinstance(option_type, OptionType):
        raise ValidationError(f"Invalid option type: {option_type}")

    # Sanity check: option price shouldn't exceed spot by too much
    if market_price > spot * 1.5:
        raise ValidationError(
            f"Option price {market_price} seems too high relative to spot {spot}"
        )

    return option_type


def get_option_flag(option_type: OptionType) -> str:
    """Get option flag for calculations ('c' or 'p')"""
    if option_type in [OptionType.CALL, OptionType.CE]:
        return 'c'
    else:
        return 'p'


def calculate_moneyness(spot: float, strike: float) -> float:
    """Calculate moneyness (spot/strike)"""
    return spot / strike if strike > 0 else 0


def is_near_expiry(hours_to_expiry: float) -> bool:
    """Check if option is near expiry (< 2 hours)"""
    return hours_to_expiry < 2.0


def is_deep_itm(spot: float, strike: float, option_type: OptionType,
                threshold: float = 0.05) -> bool:
    """Check if option is deep in the money"""
    moneyness = calculate_moneyness(spot, strike)

    if option_type in [OptionType.CALL, OptionType.CE]:
        return moneyness > (1 + threshold)
    else:  # Put
        return moneyness < (1 - threshold)


def is_deep_otm(spot: float, strike: float, option_type: OptionType,
                threshold: float = 0.05) -> bool:
    """Check if option is deep out of the money"""
    moneyness = calculate_moneyness(spot, strike)

    if option_type in [OptionType.CALL, OptionType.CE]:
        return moneyness < (1 - threshold)
    else:  # Put
        return moneyness > (1 + threshold)


def bound_implied_volatility(iv: float, hours_to_expiry: float,
                             asset_class: AssetClass = AssetClass.EQUITY) -> float:
    """
    Bound implied volatility to reasonable range based on asset class

    Args:
        iv: Raw implied volatility
        hours_to_expiry: Time to expiry in hours
        asset_class: EQUITY or CRYPTO

    Returns:
        Bounded implied volatility
    """
    if asset_class == AssetClass.CRYPTO:
        # Crypto bounds - wider range
        min_iv = 0.20  # 20%
        max_iv = 3.0   # 300%

        # Adjust bounds based on time to expiry
        if hours_to_expiry < 1:
            # Near expiry can have extreme IVs
            min_iv = 0.30
            max_iv = 5.0
        elif hours_to_expiry < 24:
            # Same day
            min_iv = 0.25
            max_iv = 4.0

    else:  # EQUITY
        # Base bounds
        min_iv = 0.05  # 5%
        max_iv = 2.0  # 200%

        # Adjust bounds based on time to expiry
        if hours_to_expiry < 1:
            # Near expiry can have extreme IVs
            min_iv = 0.10
            max_iv = 5.0
        elif hours_to_expiry < 6.25:
            # Intraday
            min_iv = 0.08
            max_iv = 3.0

    return np.clip(iv, min_iv, max_iv)


def calculate_forward_price(spot: float, risk_free_rate: float,
                            dividend_yield: float, time_to_expiry: float) -> float:
    """Calculate forward price using cost of carry"""
    return spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)


def get_trading_days_between(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of trading days between two dates"""
    if start_date >= end_date:
        return 0

    days = 0
    current = start_date.date()
    end = end_date.date()

    while current <= end:
        if current.weekday() not in [5, 6]:  # Not weekend
            days += 1
        current += timedelta(days=1)

    return days