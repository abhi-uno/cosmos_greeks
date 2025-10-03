"""
Volatility models and implied volatility calculations
"""

import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm
from typing import Optional, Tuple
import warnings

from .models import OptionType, AssetClass
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

    def __init__(self, asset_class: AssetClass):
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.asset_class = asset_class

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
                # Convert hours based on time_to_expiry already being in years
                hours_to_expiry = time_to_expiry * 365.25 * 24
                return bound_implied_volatility(iv, hours_to_expiry, self.asset_class)

        except Exception:
            pass

        # Fallback to Brent's method (more robust)
        try:
            iv = self._brent_iv(
                spot, strike, time_to_expiry, market_price,
                option_type, risk_free_rate
            )
            hours_to_expiry = time_to_expiry * 365.25 * 24
            return bound_implied_volatility(iv, hours_to_expiry, self.asset_class)

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
            initial_guess = (market_price / spot) * np.sqrt(2 * np.pi / time_to_expiry)
        else:
            # General approximation
            moneyness = np.log(spot / strike)
            normalized_price = market_price / spot

            # Adjusted for moneyness
            initial_guess = normalized_price * np.sqrt(2 * np.pi / time_to_expiry)
            initial_guess *= (1 + 0.25 * moneyness ** 2)

        # Adjust initial guess based on asset class
        if self.asset_class == AssetClass.CRYPTO:
            # Crypto typically has higher vol, adjust initial guess
            initial_guess = np.clip(initial_guess, 0.3, 3.0)
        else:
            initial_guess = np.clip(initial_guess, 0.1, 2.0)

        return initial_guess

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

            # Newton step with damping for stability
            vol_change = price_diff / vega

            # Damping factor for large changes
            if abs(vol_change) > 0.5:
                vol_change = 0.5 * np.sign(vol_change)

            vol -= vol_change

            # Keep vol positive and within reasonable bounds
            if self.asset_class == AssetClass.CRYPTO:
                vol = max(vol, 0.1)  # Higher minimum for crypto
                if vol > 5.0:  # Higher maximum for crypto
                    return None
            else:
                vol = max(vol, 0.001)
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

        # Set bounds based on asset class
        if self.asset_class == AssetClass.CRYPTO:
            vol_low = 0.1   # Higher lower bound for crypto
            vol_high = 5.0  # Higher upper bound for crypto
        else:
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

        if self.asset_class == AssetClass.CRYPTO:
            # Crypto volatility levels - much higher than equity
            if 0.97 < moneyness < 1.03:
                # ATM
                base_vol = 0.60
            elif 0.93 < moneyness < 1.07:
                # Near ATM
                base_vol = 0.70
            else:
                # OTM/ITM
                base_vol = 0.80

            # Adjust for time to expiry (time_to_expiry is already in years)
            if time_to_expiry < 1 / 365:  # Less than 1 day
                vol_mult = 1.3
            elif time_to_expiry < 7 / 365:  # Less than 1 week
                vol_mult = 1.1
            else:
                vol_mult = 1.0

        else:  # EQUITY
            # Base volatility
            if 0.97 < moneyness < 1.03:
                # ATM
                base_vol = 0.15
            elif 0.93 < moneyness < 1.07:
                # Near ATM
                base_vol = 0.20
            else:
                # OTM/ITM
                base_vol = 0.25

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
                 rho: float = -0.3, nu: float = 0.4,
                 asset_class: AssetClass = AssetClass.EQUITY):
        """
        Initialize SABR parameters

        Args:
            alpha: Initial volatility
            beta: CEV exponent (0.5 for normal SABR)
            rho: Correlation between asset and volatility
            nu: Volatility of volatility
            asset_class: EQUITY or CRYPTO
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.asset_class = asset_class

        # Adjust default parameters for crypto
        if self.asset_class == AssetClass.CRYPTO:
            self.alpha = alpha if alpha != 0.25 else 0.60  # Higher base vol
            self.nu = nu if nu != 0.4 else 0.8  # Higher vol of vol

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

        hours_to_expiry = time_to_expiry * 365.25 * 24
        return bound_implied_volatility(iv, hours_to_expiry, self.asset_class)

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

        # Avoid numerical issues
        if abs(z) < 1e-10:
            x = 1.0
        else:
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

        if abs(log_fk) < 1e-10:
            # ATM case
            return self._sabr_atm_vol(forward, time_to_expiry)

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

        if abs(v) < 1e-10:
            # ATM case
            return self.alpha

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
                 sigma: float = 0.1, rho: float = -0.3, m: float = 0.0,
                 asset_class: AssetClass = AssetClass.EQUITY):
        """
        Initialize SVI parameters

        Args:
            a: Level parameter
            b: Angle parameter
            sigma: Smoothness parameter
            rho: Rotation parameter
            m: Translation parameter
            asset_class: EQUITY or CRYPTO
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.rho = rho
        self.m = m
        self.asset_class = asset_class

        # Adjust default parameters for crypto
        if self.asset_class == AssetClass.CRYPTO:
            self.a = a if a != 0.04 else 0.36  # Higher base variance
            self.b = b if b != 0.1 else 0.3    # Steeper smile

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

        hours_to_expiry = time_to_expiry * 365.25 * 24
        return bound_implied_volatility(iv, hours_to_expiry, self.asset_class)


def create_volatility_calculator(model: str = 'BLACK_SCHOLES',
                                 asset_class: AssetClass = AssetClass.EQUITY,
                                 **params) -> VolatilityCalculator:
    """
    Factory function to create volatility calculator

    Args:
        model: Model name ('BLACK_SCHOLES', 'SABR', 'SVI')
        asset_class: EQUITY or CRYPTO
        **params: Model-specific parameters

    Returns:
        VolatilityCalculator instance
    """
    model = model.upper()

    if model == 'BLACK_SCHOLES':
        return BlackScholesVolatility(asset_class=asset_class)
    elif model == 'SABR':
        return SABRVolatility(asset_class=asset_class, **params)
    elif model == 'SVI':
        return SVIVolatility(asset_class=asset_class, **params)
    else:
        raise ValueError(f"Unknown volatility model: {model}")