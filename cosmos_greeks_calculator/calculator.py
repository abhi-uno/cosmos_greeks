"""
Main calculator module for Greeks calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Optional
import warnings
import QuantLib as ql

from .models import (
    GreeksResult, OptionType, VolatilityModel,
    MarketData, PositionGreeks, PortfolioGreeks
)
from .errors import (
    GreeksCalculationError, ValidationError,
    VolatilityCalculationError
)
from .utils import (
    calculate_hours_to_expiry, validate_inputs,
    LOT_SIZES, STRIKE_INTERVALS, get_option_flag
)
from .engines import select_engine, DigitalGreeksEngine
from .volatility import create_volatility_calculator


# Module-level function for simple usage
def calculate_greeks(spot: float,
                     strike: float,
                     expiry_datetime: datetime,
                     market_price: float,
                     option_type: Union[str, OptionType],
                     underlying: str = 'NIFTY',
                     risk_free_rate: float = 0.0,
                     volatility_model: str = 'BLACK_SCHOLES',
                     current_datetime: Optional[datetime] = None,
                     **volatility_params) -> GreeksResult:
    """
    Calculate option Greeks (stateless function)

    Args:
        spot: Current spot price
        strike: Option strike price
        expiry_datetime: Option expiry date and time
        market_price: Current market price of option
        option_type: 'CE'/'PE' or OptionType enum
        underlying: Underlying instrument ('NIFTY', 'SENSEX', etc.)
        risk_free_rate: Risk-free interest rate (default: 0.0)
        volatility_model: Volatility model to use
        current_datetime: Current date and time (defaults to now)
        **volatility_params: Additional parameters for volatility model

    Returns:
        GreeksResult object with calculated Greeks

    Example:
        import cosmos_greeks_calculator as cgc
        from datetime import datetime

        result = cgc.calculate_greeks(
            spot=25000,
            strike=25100,
            expiry_datetime=datetime(2025, 7, 10, 15, 30),
            market_price=45.5,
            option_type='CE'
        )

        print(f"Delta: {result.delta}")
        print(f"Gamma: {result.gamma}")
        print(f"Theta: {result.theta}")
        print(f"Vega: {result.vega}")
    """
    calculator = GreeksCalculator(
        underlying=underlying,
        risk_free_rate=risk_free_rate,
        volatility_model=volatility_model,
        **volatility_params
    )

    return calculator.calculate(
        spot=spot,
        strike=strike,
        expiry_datetime=expiry_datetime,
        market_price=market_price,
        option_type=option_type,
        current_datetime=current_datetime
    )


class GreeksCalculator:
    """
    Thread-safe Greeks calculator for production use

    Each thread should create its own instance to avoid QuantLib state conflicts.

    Attributes:
        underlying: Underlying instrument name
        risk_free_rate: Risk-free interest rate
        volatility_model: Volatility model to use

    Example:
        calculator = GreeksCalculator(underlying='NIFTY')

        # Single calculation
        result = calculator.calculate(
            spot=25000,
            strike=25100,
            expiry_datetime=datetime(2025, 7, 10, 15, 30),
            market_price=45.5,
            option_type='CE'
        )

        # Batch calculation
        results = calculator.calculate_batch(positions_df)
    """

    def __init__(self,
                 underlying: str = 'NIFTY',
                 risk_free_rate: float = 0.0,
                 volatility_model: str = 'BLACK_SCHOLES',
                 **volatility_params):
        """
        Initialize the Greeks calculator

        Args:
            underlying: Underlying instrument ('NIFTY', 'SENSEX', etc.)
            risk_free_rate: Risk-free interest rate (annual)
            volatility_model: Model for IV calculation
            **volatility_params: Parameters for volatility model
        """
        self.underlying = underlying.upper()
        self.risk_free_rate = risk_free_rate
        self.volatility_model = volatility_model.upper()
        self.volatility_params = volatility_params

        # Validate underlying
        if self.underlying not in LOT_SIZES:
            warnings.warn(f"Unknown underlying {self.underlying}, using NIFTY defaults")
            self.lot_size = 50
            self.strike_interval = 50
        else:
            self.lot_size = LOT_SIZES[self.underlying]
            self.strike_interval = STRIKE_INTERVALS[self.underlying]

        # Setup QuantLib for this instance
        self._setup_quantlib()

        # Create volatility calculator
        self.vol_calculator = create_volatility_calculator(
            self.volatility_model, **self.volatility_params
        )

    def _setup_quantlib(self):
        """Setup QuantLib for this thread/instance"""
        self.evaluation_date = ql.Date.todaysDate()
        # Note: QuantLib.Settings is global, so be careful in multi-threaded environments
        ql.Settings.instance().evaluationDate = self.evaluation_date

    def _calculate_adjusted_price(self, spot: float, strike: float,
                                  market_price: float, option_type: OptionType,
                                  hours_to_expiry: float) -> tuple:
        """
        Calculate adjusted price when market price is below intrinsic value

        Returns:
            tuple: (adjusted_price, adjustment_made, original_price, intrinsic_value)
        """
        # Calculate intrinsic value
        if option_type in [OptionType.CALL, OptionType.CE]:
            intrinsic = max(0, spot - strike)
        else:  # Put
            intrinsic = max(0, strike - spot)

        # Check if adjustment needed (with 5% tolerance)
        if market_price < intrinsic * 0.95:
            # Calculate premium based on time to expiry
            days_to_expiry = hours_to_expiry / 6.25  # Convert to trading days

            if days_to_expiry <= 7:
                premium_pct = 0.02  # 2%
            elif days_to_expiry <= 30:
                premium_pct = 0.03  # 3%
            else:
                premium_pct = 0.05  # 5%

            # Calculate adjusted price
            adjusted_price = intrinsic * (1 + premium_pct)
            # Ensure minimum absolute premium of 0.05
            adjusted_price = max(adjusted_price, intrinsic + 0.05)

            return adjusted_price, True, market_price, intrinsic

        return market_price, False, market_price, intrinsic

    def calculate(self,
                  spot: float,
                  strike: float,
                  expiry_datetime: datetime,
                  market_price: float,
                  option_type: Union[str, OptionType],
                  current_datetime: Optional[datetime] = None) -> GreeksResult:
        """
        Calculate Greeks for a single option

        Args:
            spot: Current spot price
            strike: Option strike price
            expiry_datetime: Option expiry date and time
            market_price: Current market price of option
            option_type: Option type ('CE'/'PE' or OptionType)
            current_datetime: Current date and time (defaults to now)
        Returns:
            GreeksResult object
        """
        try:
            # Basic input validation (without intrinsic check)
            if spot <= 0:
                return GreeksResult.create_invalid(f"Spot price must be positive, got {spot}")
            if strike <= 0:
                return GreeksResult.create_invalid(f"Strike price must be positive, got {strike}")
            if market_price < 0:
                return GreeksResult.create_invalid(f"Market price cannot be negative, got {market_price}")

            # Convert option type
            if isinstance(option_type, str):
                try:
                    option_type = OptionType.from_string(option_type)
                except ValueError as e:
                    return GreeksResult.create_invalid(str(e))
            elif not isinstance(option_type, OptionType):
                return GreeksResult.create_invalid(f"Invalid option type: {option_type}")

            # Calculate hours to expiry
            hours_to_expiry = calculate_hours_to_expiry(expiry_datetime, current_datetime)

            # Handle expired options
            if hours_to_expiry <= 0:
                return self._handle_expired_option(
                    spot, strike, market_price, option_type
                )

            # Check and adjust for intrinsic violations
            adjusted_price, was_adjusted, original_price, intrinsic_value = self._calculate_adjusted_price(
                spot, strike, market_price, option_type, hours_to_expiry
            )

            # Convert to time in years for calculations
            # CRITICAL: hours_to_expiry is in TRADING hours, not calendar hours
            # There are 252 trading days per year, 6.25 hours per day
            time_to_expiry = hours_to_expiry / (252 * 6.25)

            # Calculate implied volatility with adjusted price
            try:
                implied_vol = self.vol_calculator.get_implied_volatility(
                    spot=spot,
                    strike=strike,
                    time_to_expiry=time_to_expiry,
                    market_price=adjusted_price,  # Use adjusted price
                    option_type=option_type,
                    risk_free_rate=self.risk_free_rate
                )
            except Exception as e:
                # Use fallback volatility
                implied_vol = self._estimate_fallback_volatility(
                    spot, strike, hours_to_expiry, option_type
                )
                # Don't warn for every calculation, just use the fallback

            # Select appropriate engine
            engine = select_engine(hours_to_expiry)

            # Calculate Greeks
            greeks = engine.calculate(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=implied_vol,
                risk_free_rate=self.risk_free_rate,
                option_type=option_type
            )

            # Special handling for near-expiry theta ONLY for digital engine
            if hours_to_expiry < 0.5 and isinstance(engine, DigitalGreeksEngine):
                # For digital approximation, theta is just remaining premium
                greeks['theta'] = -market_price

            # Get method name
            method_map = {
                'DigitalGreeksEngine': 'digital',
                'AnalyticalGreeksEngine': 'analytical',
                'ScaledVolatilityEngine': 'quantlib_scaled',
                'QuantLibGreeksEngine': 'quantlib_standard'
            }
            method_name = method_map.get(engine.__class__.__name__, 'unknown')

            # Append adjustment info to method name if adjusted
            if was_adjusted:
                method_name = f"{method_name}_intrinsic_adjusted"

            # Create result
            result = GreeksResult(
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega'],
                rho=greeks.get('rho', 0.0),
                calculation_method=method_name,
                implied_volatility=implied_vol,
                hours_to_expiry=hours_to_expiry,
                is_valid=not was_adjusted  # False if adjustment was made
            )

            # Add warnings
            if was_adjusted:
                warning_msg = (f"Price below intrinsic - adjusted from {original_price:.2f} "
                               f"to {adjusted_price:.2f} (intrinsic={intrinsic_value:.2f})")
                result.warnings.append(warning_msg)

            # Add other warnings if needed
            if hours_to_expiry < 1 and abs(greeks['gamma']) > 0.1:
                result.warnings.append("High gamma near expiry - pin risk")

            return result

        except Exception as e:
            return GreeksResult.create_invalid(
                f"Calculation error: {str(e)}",
                partial_results={'hours_to_expiry': hours_to_expiry if 'hours_to_expiry' in locals() else 0}
            )

    def calculate_vectorized(self,
                             spots: np.ndarray,
                             strikes: np.ndarray,
                             expiry_datetimes: np.ndarray,
                             market_prices: np.ndarray,
                             option_types: np.ndarray,
                             current_datetimes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Vectorized Greeks calculation for performance

        Args:
            spots: Array of spot prices
            strikes: Array of strike prices
            expiry_datetimes: Array of expiry datetimes
            market_prices: Array of market prices
            option_types: Array of option types (1 for CE, 0 for PE)
            current_datetimes: Array of current datetimes (defaults to now for each)

        Returns:
            Dictionary with arrays for each Greek and metadata
        """
        n = len(spots)

        # If current_datetimes not provided, use None for each calculation
        if current_datetimes is None:
            current_datetimes = [None] * n

        # Initialize result arrays
        delta = np.zeros(n)
        gamma = np.zeros(n)
        theta = np.zeros(n)
        vega = np.zeros(n)
        rho = np.zeros(n)
        iv = np.zeros(n)
        is_valid = np.ones(n, dtype=bool)
        calculation_methods = [''] * n  # Track calculation method for each

        # Calculate hours to expiry for all positions using provided current_datetimes
        hours_to_expiry = np.zeros(n)
        for i in range(n):
            hours_to_expiry[i] = calculate_hours_to_expiry(
                expiry_datetimes[i],
                current_datetimes[i]  # This will use the provided datetime or default to now()
            )

        # Process each position
        for i in range(n):
            try:
                # Convert option type
                opt_type = OptionType.CE if option_types[i] == 1 else OptionType.PE

                # Calculate single Greek with current_datetime
                result = self.calculate(
                    spot=spots[i],
                    strike=strikes[i],
                    expiry_datetime=expiry_datetimes[i],
                    market_price=market_prices[i],
                    option_type=opt_type,
                    current_datetime=current_datetimes[i]  # Pass the specific datetime
                )

                # Store results
                delta[i] = result.delta
                gamma[i] = result.gamma
                theta[i] = result.theta
                vega[i] = result.vega
                rho[i] = result.rho
                iv[i] = result.implied_volatility
                is_valid[i] = result.is_valid
                calculation_methods[i] = result.calculation_method

            except Exception as e:
                is_valid[i] = False
                warnings.warn(f"Failed to calculate Greeks for position {i}: {str(e)}")

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'implied_volatility': iv,
            'hours_to_expiry': hours_to_expiry,
            'is_valid': is_valid,
            'calculation_methods': calculation_methods  # Added for tracking
        }

    def calculate_batch(self, positions: Union[pd.DataFrame, List[Dict]]) -> List[GreeksResult]:
        """
        Calculate Greeks for multiple positions

        Args:
            positions: DataFrame or list of dicts with columns/keys:
                - spot, strike, expiry_datetime, market_price, option_type
                - current_datetime (optional)

        Returns:
            List of GreeksResult objects
        """
        results = []

        # Convert to list of dicts if DataFrame
        if isinstance(positions, pd.DataFrame):
            positions = positions.to_dict('records')

        for pos in positions:
            result = self.calculate(
                spot=pos['spot'],
                strike=pos['strike'],
                expiry_datetime=pos['expiry_datetime'],
                market_price=pos['market_price'],
                option_type=pos['option_type'],
                current_datetime=pos.get('current_datetime')  # Optional
            )
            results.append(result)

        return results

    def calculate_portfolio_greeks(self, positions: List[Dict]) -> PortfolioGreeks:
        """
        Calculate portfolio-level Greeks

        Args:
            positions: List of position dictionaries with keys:
                - spot, strike, expiry_datetime, market_price, option_type
                - quantity: Position size (negative for short)
                - multiplier: Contract multiplier (optional, uses lot size)

        Returns:
            PortfolioGreeks object with aggregated Greeks
        """
        position_results = []

        # Calculate Greeks for each position
        for pos in positions:
            # Get multiplier
            multiplier = pos.get('multiplier', self.lot_size)

            # Calculate Greeks
            greeks = self.calculate(
                spot=pos['spot'],
                strike=pos['strike'],
                expiry_datetime=pos['expiry_datetime'],
                market_price=pos['market_price'],
                option_type=pos['option_type'],
                current_datetime=pos.get('current_datetime')
            )

            if greeks.is_valid:
                # Weight by position size
                quantity = pos['quantity']
                weight = quantity * multiplier

                # Create position result
                pos_greeks = PositionGreeks(
                    token=pos.get('token', 0),
                    strike=pos['strike'],
                    option_type=pos['option_type'],
                    quantity=quantity,
                    multiplier=multiplier,
                    unit_delta=greeks.delta,
                    unit_gamma=greeks.gamma,
                    unit_theta=greeks.theta,
                    unit_vega=greeks.vega,
                    position_delta=greeks.delta * weight,
                    position_gamma=greeks.gamma * abs(weight),  # Gamma always positive
                    position_theta=greeks.theta * weight,
                    position_vega=greeks.vega * weight,
                    market_price=pos['market_price'],
                    implied_volatility=greeks.implied_volatility,
                    hours_to_expiry=greeks.hours_to_expiry
                )

                position_results.append(pos_greeks)

        # Aggregate results
        if position_results:
            total_delta = sum(p.position_delta for p in position_results)
            total_gamma = sum(p.position_gamma for p in position_results)
            total_theta = sum(p.position_theta for p in position_results)
            total_vega = sum(p.position_vega for p in position_results)
            total_value = sum(p.market_price * abs(p.quantity * p.multiplier)
                              for p in position_results)
            net_premium = sum(p.market_price * p.quantity * p.multiplier
                              for p in position_results)
        else:
            total_delta = total_gamma = total_theta = total_vega = 0.0
            total_value = net_premium = 0.0

        return PortfolioGreeks(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            total_value=total_value,
            net_premium=net_premium,
            positions=position_results,
            calculation_time=datetime.now().isoformat(),
            underlying=self.underlying,
            spot_price=positions[0]['spot'] if positions else 0.0
        )

    def _handle_expired_option(self, spot: float, strike: float,
                               market_price: float, option_type: OptionType) -> GreeksResult:
        """Handle expired options"""

        # For expired options, Greeks are straightforward
        if option_type in [OptionType.CALL, OptionType.CE]:
            if spot > strike:
                # ITM call
                delta = 1.0
                value = spot - strike
            else:
                # OTM call
                delta = 0.0
                value = 0.0
        else:  # Put
            if spot < strike:
                # ITM put
                delta = -1.0
                value = strike - spot
            else:
                # OTM put
                delta = 0.0
                value = 0.0

        return GreeksResult(
            delta=delta,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            calculation_method='expired',
            implied_volatility=0.0,
            hours_to_expiry=0.0,
            is_valid=True,
            warnings=['Option has expired']
        )

    def _estimate_fallback_volatility(self, spot: float, strike: float,
                                      hours_to_expiry: float,
                                      option_type: OptionType) -> float:
        """Estimate volatility when IV calculation fails"""

        moneyness = spot / strike

        # Base volatility by moneyness
        if 0.97 < moneyness < 1.03:
            base_vol = 0.15  # ATM
        elif 0.93 < moneyness < 1.07:
            base_vol = 0.20  # Near ATM
        else:
            base_vol = 0.25  # OTM/ITM

        # Adjust for underlying
        if self.underlying == 'BANKNIFTY':
            base_vol *= 1.2
        elif self.underlying == 'FINNIFTY':
            base_vol *= 1.1

        # Adjust for time to expiry
        if hours_to_expiry < 2:
            base_vol *= 1.5
        elif hours_to_expiry < 6.25:
            base_vol *= 1.2

        return base_vol