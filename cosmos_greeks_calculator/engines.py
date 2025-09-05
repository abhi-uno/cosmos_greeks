"""
QuantLib-based calculation engines for different scenarios
"""

import numpy as np
import QuantLib as ql
from scipy.stats import norm
from typing import Tuple, Optional

from .models import OptionType, GreeksResult
from .utils import get_option_flag, is_deep_itm, is_deep_otm


class GreeksEngine:
    """Base class for Greeks calculation engines"""

    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, risk_free_rate: float,
                  option_type: OptionType) -> dict:
        """Calculate Greeks - to be implemented by subclasses"""
        raise NotImplementedError


class DigitalGreeksEngine(GreeksEngine):
    """
    Engine for near-expiry options (< 30 minutes)
    Uses digital approximation
    """

    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, risk_free_rate: float,
                  option_type: OptionType) -> dict:
        """Calculate Greeks using digital approximation"""

        moneyness = spot / strike

        if option_type in [OptionType.CALL, OptionType.CE]:
            if moneyness > 1.001:  # ITM
                delta = 1.0
                gamma = 0.0
            elif moneyness < 0.999:  # OTM
                delta = 0.0
                gamma = 0.0
            else:  # ATM
                delta = 0.5
                # Small gamma to represent pin risk
                gamma = 1.0 / (spot * 0.05 * 50)  # Based on tick size and lot
        else:  # Put
            if moneyness < 0.999:  # ITM
                delta = -1.0
                gamma = 0.0
            elif moneyness > 1.001:  # OTM
                delta = 0.0
                gamma = 0.0
            else:  # ATM
                delta = -0.5
                gamma = 1.0 / (spot * 0.05 * 50)

        # Near expiry: theta is remaining premium, vega is negligible
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': 0.0,  # Will be calculated from market price
            'vega': 0.0,
            'rho': 0.0
        }


class AnalyticalGreeksEngine(GreeksEngine):
    """
    Custom analytical engine for 0-DTE options (30 mins to 6 hours)
    Direct Black-Scholes implementation with precise time handling
    """

    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, risk_free_rate: float,
                  option_type: OptionType) -> dict:
        """Calculate Greeks analytically with exact time"""

        # Convert time to years for BS formula
        T = time_to_expiry

        # Avoid numerical issues
        if T < 1e-10:
            # Use digital approximation
            digital_engine = DigitalGreeksEngine()
            return digital_engine.calculate(
                spot, strike, time_to_expiry, volatility,
                risk_free_rate, option_type
            )

        # Standard Black-Scholes
        sqrt_T = np.sqrt(T)
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * T) / (volatility * sqrt_T)
        d2 = d1 - volatility * sqrt_T

        # Standard normal CDF and PDF
        N = norm.cdf
        n = norm.pdf

        phi_d1 = n(d1)

        # Calculate Greeks based on option type
        if option_type in [OptionType.CALL, OptionType.CE]:
            delta = N(d1)

            # Theta for calls (annual)
            theta_annual = (
                    -spot * phi_d1 * volatility / (2 * sqrt_T)
                    - risk_free_rate * strike * np.exp(-risk_free_rate * T) * N(d2)
            )
        else:  # Put
            delta = -N(-d1)

            # Theta for puts (annual)
            theta_annual = (
                    -spot * phi_d1 * volatility / (2 * sqrt_T)
                    + risk_free_rate * strike * np.exp(-risk_free_rate * T) * N(-d2)
            )

        # Common Greeks
        gamma = phi_d1 / (spot * volatility * sqrt_T)
        vega = spot * phi_d1 * sqrt_T  # Per 1% change

        # Rho calculation
        if option_type in [OptionType.CALL, OptionType.CE]:
            rho = strike * T * np.exp(-risk_free_rate * T) * N(d2) / 100
        else:
            rho = -strike * T * np.exp(-risk_free_rate * T) * N(-d2) / 100

        # FIX: For 0DTE options, return total theta until expiry
        # For regular options, convert annual theta to daily
        if T < 1 / 365.25:  # Less than 1 day
            # For 0DTE, theta is total until expiry
            theta = theta_annual * T
        else:
            # For regular options, convert to daily
            theta = theta_annual / 365.25

        # Apply bounds for extreme cases
        gamma = self._bound_gamma(gamma, spot, strike, time_to_expiry * 365.25 * 24)

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'd1': d1,
            'd2': d2
        }

    def _bound_gamma(self, gamma: float, spot: float, strike: float,
                     hours_to_expiry: float) -> float:
        """Bound gamma to practical limits"""

        # Maximum practical gamma based on tick size and lot size
        tick_size = 0.05
        lot_size = 50  # Default NIFTY
        max_practical_gamma = 1.0 / (spot * tick_size * lot_size)

        # For very near expiry ATM options, gamma can explode
        moneyness = spot / strike
        if 0.99 < moneyness < 1.01 and hours_to_expiry < 1:
            # Use practical maximum
            return min(gamma, max_practical_gamma)

        return gamma


class QuantLibGreeksEngine(GreeksEngine):
    """
    QuantLib-based engine for standard options (> 6 hours)
    Uses QuantLib's analytical European engine
    """

    def __init__(self):
        """Initialize QuantLib settings"""
        self.evaluation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = self.evaluation_date
        self.day_counter = ql.Actual365Fixed()
        self.calendar = ql.NullCalendar()

    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, risk_free_rate: float,
                  option_type: OptionType) -> dict:
        """Calculate Greeks using QuantLib"""

        # Convert to QuantLib types
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

        # Create flat term structures
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.evaluation_date, risk_free_rate, self.day_counter)
        )
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.evaluation_date, 0.0, self.day_counter)
        )
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.evaluation_date, self.calendar,
                                volatility, self.day_counter)
        )

        # Create process
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_ts, flat_ts, flat_vol_ts
        )

        # Create option
        days_to_expiry = int(time_to_expiry * 365.25)
        expiry_date = self.evaluation_date + max(1, days_to_expiry)

        exercise = ql.EuropeanExercise(expiry_date)

        if option_type in [OptionType.CALL, OptionType.CE]:
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        else:
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)

        option = ql.VanillaOption(payoff, exercise)

        # Set pricing engine
        engine = ql.AnalyticEuropeanEngine(bsm_process)
        option.setPricingEngine(engine)

        # Calculate Greeks
        try:
            delta = option.delta()
            gamma = option.gamma()
            theta_annual = option.theta()  # QuantLib returns ANNUAL theta
            vega = option.vega()
            rho = option.rho() / 100

            # CRITICAL: QuantLib's theta is annual, convert to daily
            theta_daily = theta_annual / 365.25

            # Handle 0DTE vs standard options
            if time_to_expiry < 1.0 / 365.25:  # Less than 1 day
                # For 0DTE, return total theta until expiry
                theta = theta_annual * time_to_expiry
            else:
                # For standard options, return daily theta
                theta = theta_daily

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }

        except Exception as e:
            # Fallback to analytical engine
            analytical = AnalyticalGreeksEngine()
            return analytical.calculate(
                spot, strike, time_to_expiry, volatility,
                risk_free_rate, option_type
            )

class ScaledVolatilityEngine(GreeksEngine):
    """
    Engine for intraday options using QuantLib with scaled volatility
    Handles the time-to-expiry issue by scaling volatility
    """

    def __init__(self):
        """Initialize QuantLib settings"""
        self.evaluation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = self.evaluation_date
        self.day_counter = ql.Actual365Fixed()
        self.calendar = ql.NullCalendar()

    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, risk_free_rate: float,
                  option_type: OptionType) -> dict:
        """Calculate Greeks using scaled volatility approach"""

        # Set artificial expiry far in future
        artificial_time = 1.0  # 1 year
        artificial_days = 365

        # Scale volatility to maintain correct option prices
        # Key insight: BS price depends on σ√T
        time_scaling_factor = artificial_time / time_to_expiry
        scaled_volatility = volatility / np.sqrt(time_scaling_factor)

        # Create QuantLib objects
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.evaluation_date, risk_free_rate, self.day_counter)
        )
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.evaluation_date, 0.0, self.day_counter)
        )
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.evaluation_date, self.calendar,
                                scaled_volatility, self.day_counter)
        )

        # Create process
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_ts, flat_ts, flat_vol_ts
        )

        # Create option with artificial expiry
        expiry_date = self.evaluation_date + artificial_days
        exercise = ql.EuropeanExercise(expiry_date)

        if option_type in [OptionType.CALL, OptionType.CE]:
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        else:
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)

        option = ql.VanillaOption(payoff, exercise)

        # Set pricing engine
        engine = ql.AnalyticEuropeanEngine(bsm_process)
        option.setPricingEngine(engine)

        # Calculate Greeks and scale back
        try:
            delta = option.delta()
            gamma = option.gamma() / time_scaling_factor
            theta_annual_artificial = option.theta()  # Annual theta for the artificial 1-year option
            vega_artificial = option.vega()
            rho = option.rho() * time_to_expiry / 100

            # Fix theta calculation
            # The artificial option has different time, so we need to adjust
            # QuantLib gives us annual theta for a 1-year option
            # We need to scale it back to our actual option

            # First get daily theta for the artificial option
            theta_daily_artificial = theta_annual_artificial / 365.25

            # The relationship between thetas is based on time scaling
            # Since we scaled volatility by 1/sqrt(time_scaling_factor),
            # theta scales differently
            theta_daily_actual = theta_daily_artificial / time_scaling_factor

            # For intraday, return total theta until expiry
            theta = theta_daily_actual * time_to_expiry * 365.25

            # Fix vega calculation
            # Vega also needs adjustment because we scaled volatility
            # When we scale vol by 1/sqrt(k), vega scales by sqrt(k)
            vega = vega_artificial * np.sqrt(time_scaling_factor)

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }

        except Exception as e:
            # Fallback to analytical engine
            analytical = AnalyticalGreeksEngine()
            return analytical.calculate(
                spot, strike, time_to_expiry, volatility,
                risk_free_rate, option_type
            )


def select_engine(hours_to_expiry: float) -> GreeksEngine:
    """
    Select appropriate engine based on time to expiry

    Args:
        hours_to_expiry: Hours remaining to expiry

    Returns:
        Appropriate GreeksEngine instance
    """
    if hours_to_expiry < 0.5:
        # Less than 30 minutes - digital approximation
        return DigitalGreeksEngine()
    elif hours_to_expiry < 6.25:
        # Less than 1 trading day - use analytical engine
        # This is more accurate than scaled volatility for intraday
        return AnalyticalGreeksEngine()
    else:
        # Standard - pure QuantLib
        return QuantLibGreeksEngine()