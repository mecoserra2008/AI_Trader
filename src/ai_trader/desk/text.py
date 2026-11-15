"""Ticker extraction and a small crypto tone lexicon (VADER-style compound score, no models, no downloads).

tone(text) -> [-1, 1]: sum of word valences (negation in the 3 preceding tokens flips by -0.74, a booster in the 2
preceding tokens scales by 1.3), squashed by s / sqrt(s^2 + 15) as in VADER. Multi-word phrases are matched first.
tickers(text, index) -> sorted bases: "$BASE" or an upper-case "BASE" word (bases of >= 3 letters that are not common
English words), plus coin names (longest alias first, so "bitcoin cash" is BCH and not BTC).
"""

from __future__ import annotations

import math
import re

LEXICON: dict[str, float] = {
    # positive
    "bullish": 2.5, "bull": 1.5, "bulls": 1.5, "moon": 2.0, "mooning": 2.5, "pump": 1.5, "pumping": 1.8, "rally": 2.0, "rallies": 2.0,
    "rallied": 2.0, "surge": 2.2, "surges": 2.2, "surged": 2.2, "soar": 2.5, "soars": 2.5, "soared": 2.5, "jump": 1.2, "jumps": 1.2,
    "gain": 1.5, "gains": 1.5, "gained": 1.5, "breakout": 1.8, "rebound": 1.5, "rebounds": 1.5, "recover": 1.3, "recovers": 1.3,
    "recovery": 1.3, "green": 1.0, "ath": 2.2, "record": 1.2, "high": 0.5, "higher": 0.8, "up": 0.4, "buy": 1.0, "buying": 1.0,
    "accumulate": 1.3, "accumulating": 1.3, "long": 0.6, "adoption": 1.8, "approve": 1.8, "approved": 2.0, "approval": 1.8,
    "inflows": 1.8, "inflow": 1.8, "partnership": 1.5, "upgrade": 1.3, "launch": 0.8, "launches": 0.8, "strong": 1.5,
    "strength": 1.2, "support": 0.5, "win": 1.5, "wins": 1.5, "profit": 1.5, "profits": 1.5, "optimistic": 2.0, "optimism": 2.0,
    "hope": 1.0, "good": 1.5, "great": 2.0, "amazing": 2.5, "love": 2.0, "best": 2.0, "positive": 1.5, "boost": 1.5, "boosts": 1.5,
    "outperform": 1.5, "outperforms": 1.5, "undervalued": 1.3, "hodl": 1.0, "wagmi": 1.8, "lfg": 2.0, "rich": 1.5, "beat": 1.0,
    "beats": 1.0, "upside": 1.3, "rip": 0.0, "reclaim": 1.3, "reclaims": 1.3, "bounce": 1.2, "bounces": 1.2,
    # negative
    "bearish": -2.5, "bear": -1.5, "bears": -1.5, "dump": -2.0, "dumping": -2.2, "dumped": -2.0, "crash": -3.0, "crashes": -3.0,
    "crashed": -3.0, "plunge": -2.8, "plunges": -2.8, "plunged": -2.8, "tank": -2.2, "tanks": -2.2, "tanked": -2.2, "drop": -1.5,
    "drops": -1.5, "dropped": -1.5, "fall": -1.5, "falls": -1.5, "fell": -1.5, "slump": -2.2, "slumps": -2.2, "sink": -1.8,
    "sinks": -1.8, "slide": -1.5, "slides": -1.5, "decline": -1.5, "declines": -1.5, "loss": -1.8, "losses": -1.8, "lose": -1.5,
    "losing": -1.5, "lost": -1.5, "red": -1.0, "sell": -1.0, "selling": -1.2, "sold": -0.8, "short": -0.6, "shorts": -0.4,
    "rekt": -2.5, "liquidated": -2.2, "liquidation": -1.8, "liquidations": -1.8, "hack": -3.0, "hacked": -3.0, "exploit": -3.0,
    "exploited": -3.0, "scam": -3.0, "fraud": -3.0, "rug": -3.0, "lawsuit": -2.0, "sues": -2.0, "sued": -2.0, "ban": -2.2,
    "banned": -2.2, "fud": -1.5, "fear": -2.0, "panic": -2.8, "capitulation": -2.5, "delist": -2.5, "delisted": -2.5,
    "delisting": -2.5, "outage": -2.0, "halt": -1.8, "halted": -1.8, "weak": -1.5, "weakness": -1.5, "bad": -1.8, "worst": -2.5,
    "terrible": -2.5, "hate": -2.2, "negative": -1.5, "risk": -0.6, "risky": -1.0, "warning": -1.5, "warns": -1.5, "bubble": -1.5,
    "overvalued": -1.3, "outflows": -1.8, "outflow": -1.8, "down": -0.5, "lower": -0.8, "low": -0.4, "ngmi": -1.8, "bleed": -2.0,
    "bleeding": -2.0, "underperform": -1.5, "downside": -1.3, "worry": -1.5, "worried": -1.5, "concern": -1.2, "concerns": -1.2,
    "selloff": -2.2, "rugpull": -3.0, "insolvent": -3.0, "bankrupt": -3.0, "bankruptcy": -3.0, "investigation": -1.5,
}
PHRASES: dict[str, float] = {
    "all time high": 2.2, "to the moon": 2.5, "short squeeze": 1.5, "golden cross": 1.5, "buy the dip": 1.0, "sell off": -2.2,
    "rug pull": -3.0, "death cross": -1.8, "bear market": -2.0, "bull market": 2.0, "dead cat bounce": -1.5, "price target cut": -1.5,
}
NEGATIONS = frozenset({"not", "no", "never", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "can't", "cannot",
                       "won't", "wouldn't", "shouldn't", "without", "nor", "hardly", "isnt", "dont", "doesnt", "didnt", "cant", "wont"})
BOOSTERS = frozenset({"very", "extremely", "massively", "hugely", "super", "really", "so", "incredibly", "insanely", "totally", "absolutely", "huge", "massive"})
ALPHA = 15.0

ALIASES: dict[str, str] = {
    "bitcoin cash": "BCH", "bitcoin": "BTC", "ethereum classic": "ETC", "ethereum": "ETH", "ether": "ETH", "solana": "SOL",
    "ripple": "XRP", "dogecoin": "DOGE", "cardano": "ADA", "chainlink": "LINK", "litecoin": "LTC", "avalanche": "AVAX",
    "polkadot": "DOT", "zcash": "ZEC", "hyperliquid": "HYPE", "near protocol": "NEAR", "pepe": "PEPE", "binance coin": "BNB",
    "tron": "TRX", "stellar": "XLM", "monero": "XMR", "aave": "AAVE", "uniswap": "UNI", "arbitrum": "ARB", "aptos": "APT",
    "hedera": "HBAR", "toncoin": "TON", "ethena": "ENA", "worldcoin": "WLD", "injective": "INJ", "ondo": "ONDO", "bittensor": "TAO",
    "fartcoin": "FARTCOIN", "pump.fun": "PUMPFUN", "optimism": "OP", "jupiter": "JUP", "pengu": "PENGU", "starknet": "STRK",
    "filecoin": "FIL", "cosmos": "ATOM", "kaspa": "KAS", "bonk": "BONK", "sui network": "SUI", "microstrategy": "MSTR", "nvidia": "NVDA",
    "shiba inu": "SHIB", "polygon": "POL", "render network": "RENDER", "celestia": "TIA", "sei network": "SEI", "aster": "ASTER",
}
COMMON = frozenset({
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET",
    "HAS", "HIM", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET", "PUT", "SAY",
    "SHE", "TOO", "USE", "NEAR", "LINK", "MOVE", "GAS", "SAND", "FLOW", "ROSE", "MASK", "CAKE", "BAND", "MAGIC", "BLUR", "PEOPLE",
    "COMP", "TRUMP", "PUMP", "INIT", "ACT", "ZERO", "ONDO", "HIGH", "LOW", "BIG", "TOP", "WIN", "BEST", "REAL", "MEME", "AI",
    "GOOD", "JUST", "LIKE", "WILL", "WITH", "THIS", "THAT", "FROM", "HAVE", "BEEN", "THEY", "WHAT", "WHEN", "YOUR", "ETF", "SEC",
    "USA", "USD", "CEO", "IPO", "API", "NFT", "DAO", "FED", "CPI", "GDP", "ATH", "FUD", "OTC", "DEX", "CEX", "TVL", "APY",
    "KAVA", "STRK", "AUCTION", "HOME", "OPEN", "TRUE", "FUN", "SUN", "BOT", "SAFE", "PEOPLE", "EDGE", "ERA", "LIT", "AKE", "AT",
})
_TOK = re.compile(r"[a-z0-9$']+")


def tone(text: str | None) -> float:
    if not text:
        return 0.0
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    s = 0.0
    for ph, v in PHRASES.items():
        n = t.count(f" {ph} ")
        if n:
            s += n * v
            t = t.replace(f" {ph} ", " ")
    toks = _TOK.findall(t)
    for i, w in enumerate(toks):
        v = LEXICON.get(w)
        if not v:
            continue
        if any(x in BOOSTERS for x in toks[max(0, i - 2):i]):
            v *= 1.3
        if any(x in NEGATIONS for x in toks[max(0, i - 3):i]):
            v *= -0.74
        s += v
    return s / math.sqrt(s * s + ALPHA)


class TickerIndex:
    def __init__(self, bases: list[str] | set[str], aliases: dict[str, str] | None = None):
        self.bases = {b.upper() for b in bases}
        al = {k: v for k, v in (aliases or ALIASES).items() if v in self.bases}
        self.aliases = sorted(al.items(), key=lambda kv: -len(kv[0]))
        self._alias_re = [(re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])"), v) for k, v in self.aliases]
        upper = sorted((b for b in self.bases if len(b) >= 3 and b not in COMMON and b.isalnum()), key=len, reverse=True)
        self._upper_re = re.compile(r"(?<![A-Za-z0-9$])(" + "|".join(map(re.escape, upper)) + r")(?![A-Za-z0-9])") if upper else None
        cash = sorted((b for b in self.bases if b.isalnum()), key=len, reverse=True)
        self._cash_re = re.compile(r"\$(" + "|".join(map(re.escape, cash)) + r")(?![A-Za-z0-9])", re.IGNORECASE) if cash else None

    def find(self, text: str | None, default: str | None = None) -> list[str]:
        out = set()
        if default and default.upper() in self.bases:
            out.add(default.upper())
        if not text:
            return sorted(out)
        if self._cash_re:
            out |= {m.upper() for m in self._cash_re.findall(text)}
        if self._upper_re:
            out |= set(self._upper_re.findall(text))
        low = text.lower()
        for rx, v in self._alias_re:
            if rx.search(low):
                out.add(v)
                low = rx.sub(" ", low)
        return sorted(out)


__all__ = ["ALIASES", "LEXICON", "TickerIndex", "tone"]
