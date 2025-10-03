"""
Calculation engines for different scenarios
"""

import numpy as np
from scipy.stats import norm
from typing import Optional

from .models import OptionType, AssetClass


class GreeksEngine:
    """Base class for Greeks calculation engines"""

    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, risk_free_rate: float,
                  option_type: OptionType, asset_class: AssetClass = AssetClass.EQUITY) -> dict:
        """Calculate Greeks - to be implemented by subclasses"""
        raise NotImplementedError


class DigitalGreeksEngine(GreeksEngine):
    """
    Engine for near-expiry options (< 30 minutes for equity, < 1 hour for crypto)
    Uses digital approximation
    """

    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, risk_free_rate: float,
                  option_type: OptionType, asset_class: AssetClass = AssetClass.EQUITY) -> dict:
        """Calculate Greeks using digital approximation"""

        moneyness = spot / strike

        # Determine lot size and tick size based on asset class
        if asset_class == AssetClass.CRYPTO:
            tick_size = 1.0  # $1 tick for crypto
            lot_size = 0.001  # BTC lot size
        else:
            tick_size = 0.05  # INR tick for equity
            lot_size = 50  # Default NIFTY lot

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
                gamma = 1.0 / (spot * tick_size * lot_size)
        else:  # Put
            if moneyness < 0.999:  # ITM
                delta = -1.0
                gamma = 0.0
            elif moneyness > 1.001:  # OTM
                delta = 0.0
                gamma = 0.0
            else:  # ATM
                delta = -0.5
                gamma = 1.0 / (spot * tick_size * lot_size)

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
    Analytical Black-Scholes engine for all non-digital options
    Direct implementation with precise time handling
    """

    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, risk_free_rate: float,
                  option_type: OptionType, asset_class: AssetClass = AssetClass.EQUITY) -> dict:
        """Calculate Greeks analytically with exact time"""

        # Convert time to years for BS formula
        T = time_to_expiry

        # Avoid numerical issues
        if T < 1e-10:
            # Use digital approximation
            digital_engine = DigitalGreeksEngine()
            return digital_engine.calculate(
                spot, strike, time_to_expiry, volatility,
                risk_free_rate, option_type, asset_class
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
        vega = spot * phi_d1 * sqrt_T / 100  # Per 1% change

        # Rho calculation
        if option_type in [OptionType.CALL, OptionType.CE]:
            rho = strike * T * np.exp(-risk_free_rate * T) * N(d2) / 100
        else:
            rho = -strike * T * np.exp(-risk_free_rate * T) * N(-d2) / 100

        # Theta conversion based on asset class
        if asset_class == AssetClass.CRYPTO:
            # For crypto, convert to daily theta
            if T < 1 / 365:  # Less than 1 day
                # For same-day expiry, return total theta until expiry
                theta = theta_annual * T
            else:
                # For multi-day, return daily theta
                theta = theta_annual / 365
        else:  # EQUITY
            # For equity, use trading days
            if T < 1 / 365.25:  # Less than 1 day
                # For 0DTE, theta is total until expiry
                theta = theta_annual * T
            else:
                # For regular options, convert to daily
                theta = theta_annual / 365.25

        # Apply bounds for extreme cases
        gamma = self._bound_gamma(gamma, spot, strike, time_to_expiry * 365.25 * 24, asset_class)

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
                     hours_to_expiry: float, asset_class: AssetClass = AssetClass.EQUITY) -> float:
        """Bound gamma to practical limits"""

        # Maximum practical gamma based on tick size and lot size
        if asset_class == AssetClass.CRYPTO:
            tick_size = 1.0  # $1 tick
            lot_size = 0.001  # BTC lot
        else:
            tick_size = 0.05
            lot_size = 50  # Default NIFTY

        max_practical_gamma = 1.0 / (spot * tick_size * lot_size)

        # For very near expiry ATM options, gamma can explode
        moneyness = spot / strike
        if 0.99 < moneyness < 1.01 and hours_to_expiry < 1:
            # Use practical maximum
            return min(gamma, max_practical_gamma)

        return gamma


def select_engine(hours_to_expiry: float, asset_class: AssetClass = AssetClass.EQUITY) -> GreeksEngine:
    """
    Select appropriate engine based on time to expiry and asset class

    Args:
        hours_to_expiry: Hours remaining to expiry
        asset_class: EQUITY or CRYPTO

    Returns:
        Appropriate GreeksEngine instance
    """
    if asset_class == AssetClass.CRYPTO:
        # Crypto thresholds
        if hours_to_expiry < 1:
            # Less than 1 hour - digital approximation
            return DigitalGreeksEngine()
        else:
            # All other cases - use analytical engine
            return AnalyticalGreeksEngine()
    else:
        # Equity thresholds
        if hours_to_expiry < 0.5:
            # Less than 30 minutes - digital approximation
            return DigitalGreeksEngine()
        else:
            # All other cases - use analytical engine
            return AnalyticalGreeksEngine()