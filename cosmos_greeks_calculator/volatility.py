"""
Volatility models and implied volatility calculations
"""

import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm
from typing import Optional, Tuple
import warnings

from .models import OptionType
from .utils import get_option_flag, bound_implied_volatility
from .errors import VolatilityCalculationError


class VolatilityCalculator:
    """Base class for volatility calculations"""

    def get_implied_volatility(self, spot: float, strike: float,
                               time_to_expiry: float, market_price: float,
                               option_type: OptionType, risk_free_rate: float = 0.0,
                               **kwargs) -> float:
        """Calculate implied volatility from market price"""
        raise NotImplementedError


class BlackScholesVolatility(VolatilityCalculator):
    """
    Standard Black-Scholes implied volatility calculator
    """

    def __init__(self):
        self.max_iterations = 100
        self.tolerance = 1e-6

    def get_implied_volatility(self, spot: float, strike: float,
                               time_to_expiry: float, market_price: float,
                               option_type: OptionType, risk_free_rate: float = 0.0,
                               **kwargs) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        with fallback to Brent's method
        """

        # Quick sanity checks
        if market_price <= 0:
            raise VolatilityCalculationError("Market price must be positive")

        if time_to_expiry <= 0:
            raise VolatilityCalculationError("Time to expiry must be positive")

        # Get intrinsic value
        if option_type in [OptionType.CALL, OptionType.CE]:
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)

        # If price is below intrinsic, something is wrong
        if market_price < intrinsic * 0.99:
            raise VolatilityCalculationError(
                f"Market price {market_price} is below intrinsic value {intrinsic}"
            )

        # Initial guess using Brenner-Subrahmanyam approximation
        initial_vol = self._initial_vol_estimate(
            spot, strike, time_to_expiry, market_price, option_type
        )

        try:
            # Try Newton-Raphson first (faster)
            iv = self._newton_raphson_iv(
                spot, strike, time_to_expiry, market_price,
                option_type, risk_free_rate, initial_vol
            )

            if iv is not None and 0.01 < iv < 10.0:
                return bound_implied_volatility(iv, time_to_expiry * 365.25 * 24)

        except Exception:
            pass

        # Fallback to Brent's method (more robust)
        try:
            iv = self._brent_iv(
                spot, strike, time_to_expiry, market_price,
                option_type, risk_free_rate
            )
            return bound_implied_volatility(iv, time_to_expiry * 365.25 * 24)

        except Exception as e:
            # Last resort: return a reasonable estimate
            return self._fallback_volatility_estimate(
                spot, strike, time_to_expiry, option_type
            )

    def _initial_vol_estimate(self, spot: float, strike: float,
                              time_to_expiry: float, market_price: float,
                              option_type: OptionType) -> float:
        """Brenner-Subrahmanyam approximation for initial IV guess"""

        # For ATM options
        if 0.95 < spot / strike < 1.05:
            return (market_price / spot) * np.sqrt(2 * np.pi / time_to_expiry)

        # General approximation
        moneyness = np.log(spot / strike)
        normalized_price = market_price / spot

        # Adjusted for moneyness
        vol_estimate = normalized_price * np.sqrt(2 * np.pi / time_to_expiry)
        vol_estimate *= (1 + 0.25 * moneyness ** 2)

        return np.clip(vol_estimate, 0.1, 2.0)

    def _newton_raphson_iv(self, spot: float, strike: float, time_to_expiry: float,
                           market_price: float, option_type: OptionType,
                           risk_free_rate: float, initial_vol: float) -> Optional[float]:
        """Newton-Raphson method for IV calculation"""

        vol = initial_vol
        flag = get_option_flag(option_type)

        for i in range(self.max_iterations):
            # Calculate option price and vega
            price, vega = self._black_scholes_price_and_vega(
                spot, strike, time_to_expiry, vol, risk_free_rate, flag
            )

            # Check convergence
            price_diff = price - market_price
            if abs(price_diff) < self.tolerance:
                return vol

            # Vega too small, can't continue
            if abs(vega) < 1e-10:
                return None

            # Newton step
            vol -= price_diff / vega

            # Keep vol positive
            vol = max(vol, 0.001)

            # Bound check
            if vol > 10.0:
                return None

        return None

    def _brent_iv(self, spot: float, strike: float, time_to_expiry: float,
                  market_price: float, option_type: OptionType,
                  risk_free_rate: float) -> float:
        """Brent's method for IV calculation"""

        flag = get_option_flag(option_type)

        def objective(vol):
            price = self._black_scholes_price(
                spot, strike, time_to_expiry, vol, risk_free_rate, flag
            )
            return price - market_price

        # Find bounds where objective changes sign
        vol_low = 0.01
        vol_high = 5.0

        # Adjust bounds if needed
        for _ in range(10):
            f_low = objective(vol_low)
            f_high = objective(vol_high)

            if f_low * f_high < 0:
                # Found valid bounds
                break

            if f_low > 0:
                # Even low vol gives price too high
                vol_low /= 2
            if f_high < 0:
                # Even high vol gives price too low
                vol_high *= 2

        # Use Brent's method
        iv = brentq(objective, vol_low, vol_high, xtol=self.tolerance)
        return iv

    def _black_scholes_price(self, spot: float, strike: float, time_to_expiry: float,
                             volatility: float, risk_free_rate: float, flag: str) -> float:
        """Calculate Black-Scholes option price"""

        sqrt_t = np.sqrt(time_to_expiry)
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t

        if flag == 'c':
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return price

    def _black_scholes_price_and_vega(self, spot: float, strike: float,
                                      time_to_expiry: float, volatility: float,
                                      risk_free_rate: float, flag: str) -> Tuple[float, float]:
        """Calculate Black-Scholes price and vega"""

        sqrt_t = np.sqrt(time_to_expiry)
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t

        phi_d1 = norm.pdf(d1)

        if flag == 'c':
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        vega = spot * phi_d1 * sqrt_t / 100  # Per 1% change

        return price, vega

    def _fallback_volatility_estimate(self, spot: float, strike: float,
                                      time_to_expiry: float, option_type: OptionType) -> float:
        """Fallback volatility estimate based on moneyness and time"""

        moneyness = spot / strike

        # Base volatility
        if 0.95 < moneyness < 1.05:
            # ATM
            base_vol = 0.20
        elif 0.90 < moneyness < 1.10:
            # Near ATM
            base_vol = 0.25
        else:
            # OTM/ITM
            base_vol = 0.30

        # Adjust for time to expiry
        if time_to_expiry < 1 / 365.25:  # Less than 1 day
            vol_mult = 1.5
        elif time_to_expiry < 7 / 365.25:  # Less than 1 week
            vol_mult = 1.2
        else:
            vol_mult = 1.0

        return base_vol * vol_mult


class SABRVolatility(VolatilityCalculator):
    """
    SABR (Stochastic Alpha Beta Rho) volatility model
    Better handles volatility smile
    """

    def __init__(self, alpha: float = 0.25, beta: float = 0.5,
                 rho: float = -0.3, nu: float = 0.4):
        """
        Initialize SABR parameters

        Args:
            alpha: Initial volatility
            beta: CEV exponent (0.5 for normal SABR)
            rho: Correlation between asset and volatility
            nu: Volatility of volatility
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def get_implied_volatility(self, spot: float, strike: float,
                               time_to_expiry: float, market_price: float,
                               option_type: OptionType, risk_free_rate: float = 0.0,
                               **kwargs) -> float:
        """Calculate implied volatility using SABR model"""

        # Forward price (simplified - no dividends)
        forward = spot * np.exp(risk_free_rate * time_to_expiry)

        # SABR implied volatility
        if abs(forward - strike) < 1e-10:
            # ATM case
            iv = self._sabr_atm_vol(forward, time_to_expiry)
        else:
            # General case
            iv = self._sabr_vol(forward, strike, time_to_expiry)

        return bound_implied_volatility(iv, time_to_expiry * 365.25 * 24)

    def _sabr_vol(self, forward: float, strike: float, time_to_expiry: float) -> float:
        """SABR volatility formula"""

        if self.beta == 1.0:
            # Lognormal SABR
            return self._sabr_lognormal(forward, strike, time_to_expiry)
        elif self.beta == 0.0:
            # Normal SABR
            return self._sabr_normal(forward, strike, time_to_expiry)
        else:
            # General SABR
            return self._sabr_general(forward, strike, time_to_expiry)

    def _sabr_general(self, forward: float, strike: float, time_to_expiry: float) -> float:
        """General SABR formula for 0 < beta < 1"""

        # Pre-calculations
        fk_beta = (forward * strike) ** ((1 - self.beta) / 2)
        log_fk = np.log(forward / strike)

        # z calculation
        z = (self.nu / self.alpha) * fk_beta * log_fk

        # x calculation
        x_numerator = np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho
        x = np.log(x_numerator / (1 - self.rho))

        # A term
        a = self.alpha / (fk_beta * (1 + (1 - self.beta) ** 2 / 24 * log_fk ** 2 +
                                     (1 - self.beta) ** 4 / 1920 * log_fk ** 4))

        # B term
        b = 1 + time_to_expiry * (
                (1 - self.beta) ** 2 * self.alpha ** 2 / (24 * fk_beta ** 2) +
                self.rho * self.beta * self.nu * self.alpha / (4 * fk_beta) +
                (2 - 3 * self.rho ** 2) * self.nu ** 2 / 24
        )

        # Final volatility
        if abs(z) < 1e-10:
            return a * b
        else:
            return a * (z / x) * b

    def _sabr_atm_vol(self, forward: float, time_to_expiry: float) -> float:
        """SABR ATM volatility (simplified)"""

        f_beta = forward ** (1 - self.beta)

        vol = self.alpha / f_beta * (
                1 + time_to_expiry * (
                (1 - self.beta) ** 2 * self.alpha ** 2 / (24 * forward ** (2 * (1 - self.beta))) +
                self.rho * self.beta * self.nu * self.alpha / (4 * f_beta) +
                (2 - 3 * self.rho ** 2) * self.nu ** 2 / 24
        )
        )

        return vol

    def _sabr_lognormal(self, forward: float, strike: float, time_to_expiry: float) -> float:
        """SABR formula for beta = 1 (lognormal)"""

        log_fk = np.log(forward / strike)
        z = (self.nu / self.alpha) * log_fk
        x = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))

        a = self.alpha
        b = 1 + time_to_expiry * (
                self.rho * self.nu * self.alpha / 4 +
                (2 - 3 * self.rho ** 2) * self.nu ** 2 / 24
        )

        if abs(z) < 1e-10:
            return a * b
        else:
            return a * (z / x) * b

    def _sabr_normal(self, forward: float, strike: float, time_to_expiry: float) -> float:
        """SABR formula for beta = 0 (normal)"""

        v = forward - strike
        z = (self.nu / self.alpha) * v
        x = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))

        a = self.alpha
        b = 1 + time_to_expiry * (
                self.rho * self.nu * self.alpha / 4 +
                (2 - 3 * self.rho ** 2) * self.nu ** 2 / 24
        )

        if abs(z) < 1e-10:
            return a * b
        else:
            return a * (z / x) * b


class SVIVolatility(VolatilityCalculator):
    """
    SVI (Stochastic Volatility Inspired) parametrization
    Good for fitting the entire volatility surface
    """

    def __init__(self, a: float = 0.04, b: float = 0.1,
                 sigma: float = 0.1, rho: float = -0.3, m: float = 0.0):
        """
        Initialize SVI parameters

        Args:
            a: Level parameter
            b: Angle parameter
            sigma: Smoothness parameter
            rho: Rotation parameter
            m: Translation parameter
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.rho = rho
        self.m = m

    def get_implied_volatility(self, spot: float, strike: float,
                               time_to_expiry: float, market_price: float,
                               option_type: OptionType, risk_free_rate: float = 0.0,
                               **kwargs) -> float:
        """Calculate implied volatility using SVI model"""

        # Log-moneyness
        k = np.log(strike / spot)

        # SVI variance
        variance = self.a + self.b * (
                self.rho * (k - self.m) +
                np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

        # Implied volatility
        iv = np.sqrt(max(variance, 0.0001) / time_to_expiry)

        return bound_implied_volatility(iv, time_to_expiry * 365.25 * 24)


def create_volatility_calculator(model: str = 'BLACK_SCHOLES', **params) -> VolatilityCalculator:
    """
    Factory function to create volatility calculator

    Args:
        model: Model name ('BLACK_SCHOLES', 'SABR', 'SVI')
        **params: Model-specific parameters

    Returns:
        VolatilityCalculator instance
    """
    model = model.upper()

    if model == 'BLACK_SCHOLES':
        return BlackScholesVolatility()
    elif model == 'SABR':
        return SABRVolatility(**params)
    elif model == 'SVI':
        return SVIVolatility(**params)
    else:
        raise ValueError(f"Unknown volatility model: {model}")