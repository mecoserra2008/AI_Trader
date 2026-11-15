# XS 2026-11-15T19:54Z (1h bars to 19:00) - context, not a signal: R-XS-2 / R-XS-3 / R-AVOID-1 / R-CARRY-1
Crypto EW basket n=46: 1h +52bp | 4h +9bp | 24h +9bp | 7d -1535bp; TradFi EW n=14: 1h -4 | 4h +32 | 24h +27 | 7d -512
rs = Delta ln(P / EW basket) in bp (sd = vol-standardised); funding ann % (90d pct); OI log %; basis 1e4 ln(mark/index)
v0 cells: R9 x16, R6 x4, R8 x3, R5 x2, R4 x2, R1 x1 (of 60; others not in lake / short)
## Strongest 24h (crypto)
| sym | sec | cell | rs1h | rs4h | rs24h (sd) | rs7d | fund ann% (pct) | OI24h | basis |
|---|---|---|---|---|---|---|---|---|---|
| EVAA | other | - | +610 | -143 | +4850 (+2.0) | +11158 | +51.1 (p70) | -0.7 | +11.1 |
| ARIA | other | - | +42 | -115 | +2670 (+1.3) | +6632 | +81.8 (p71) | -0.2 | +15.0 |
| KAVA | other | - | +358 | +127 | +1551 (+0.7) | -6273 | -658.4 (p0) | -0.1 | -148.2 |
| VVV | ai | - | -38 | +293 | +1434 (+2.5) | +1112 | +10.9 (p64) | +7.3 | -4.7 |
| JCT | other | - | -97 | +539 | +886 (+0.3) | -16540 | +10.9 (p32) | +7.0 | +10.2 |
| INIT | l1 | - | -120 | -212 | +583 (+0.2) | +634 | +1103.9 (p94) | +19.3 | -10.9 |
| AAVE | defi | R6 | +5 | +93 | +491 (+1.5) | -450 | +2.3 (p37) | +5.1 | -9.3 |
## Weakest 24h (crypto)
| sym | sec | cell | rs1h | rs4h | rs24h (sd) | rs7d | fund ann% (pct) | OI24h | basis |
|---|---|---|---|---|---|---|---|---|---|
| SOPH | gaming | - | +348 | -757 | -5588 (-2.3) | +1808 | -1999.3 (p6) | +8.4 | -64.8 |
| BARD | other | - | +39 | +206 | -1554 (-1.0) | +986 | +10.9 (p80) | -14.6 | -4.0 |
| ASTER | exchange | - | +174 | +322 | -1223 (-1.8) | -576 | -6.4 (p14) | -0.2 | -5.5 |
| ZRO | l1 | - | -35 | -21 | -996 (-1.8) | +29 | +4.9 (p18) | -2.4 | -11.5 |
| INJ | rwa | - | -59 | -102 | -563 (-1.1) | -100 | -19.0 (p3) | -1.7 | -13.3 |
| PUMPFUN | other | - | -122 | -107 | -433 (-1.2) | -1204 | +10.9 (p62) | +0.1 | -12.1 |
| APT | l1 | - | +52 | +89 | -433 (-1.5) | -1052 | -22.8 (p2) | -8.5 | -11.0 |
TradFi rs24h vs TradFi basket: SPCX +240, SOXL +207, KORU +156, SKHYNIX +92, SNDK +66, CL +58, MU -1, NVDA -7, QQQ -11, XAU -14
## Funding (last settled, ann %; pct of 90d window)
Most negative: SOPH -1999.3% (p6), KAVA -658.4% (p0), AKE -467.1% (p8), LIT -92.3% (p2), MSTR -89.8% (p2), WLD -66.4% (p0)
Most positive: INIT +1103.9% (p94), ARIA +81.8% (p71), SOXS +58.7% (p97), EVAA +51.1% (p70), BARD +10.9% (p80), STRK +10.9% (p68)
OI 24h movers: KORU +25.9% (px +1.8%), INIT +19.3% (px +5.9%), BARD -14.6% (px -15.5%), SOXS +14.2% (px -2.0%), CL +14.2% (px +0.8%), SKHYNIX +13.4% (px +1.2%), AVAX -13.1% (px -3.8%), XAG +11.8% (px -0.5%)
Basis extremes (mark vs index bp): KAVA -148.2, KORU +68.6, SOPH -64.8, SOXL +58.6, NVDA +53.6, SOXS -51.3
## Sectors (mean rs bp 4h / 24h / 7d; funding ann % median; OI24h mean)
- other (n 7): +24 / +1094 / -704; fund +10.9; OI -2.2%
- defi (n 3): +53 / +315 / -1009; fund +2.3; OI -3.2%
- ai (n 4): +136 / +181 / +3502; fund +10.9; OI -0.5%
- privacy (n 2): -24 / +176 / -1201; fund +10.9; OI -8.4%
- oracle (n 1): +8 / +102 / +10; fund -3.9; OI +0.1%
- l2 (n 2): -103 / +88 / -224; fund +10.9; OI -3.7%
- tradfi_etf (n 4): +5 / +32 / +316; fund +0.0; OI +9.3%
- payments (n 3): -7 / -4 / +139; fund +1.2; OI -2.9%
- tradfi_commodity (n 3): -13 / -10 / -40; fund +0.0; OI +9.6%
- tradfi_stock (n 7): +3 / -14 / -163; fund +0.0; OI +1.2%
- btc (n 1): -13 / -135 / -62; fund -1.4; OI -1.6%
- l1 (n 12): +11 / -148 / -218; fund -7.3; OI -3.0%
## v3 context (xs3-20261115T19, 50 min old; low-vol ranking, not a signal, R-XS-2/3)
4h rel median top: BTC +3, DOGE -0, ETH -1, BNB -1, XRP -2 | bottom: HYPE -6, UNI -9, ENA -12, NEAR -14, AAVE -15
## Flags (28; rules {'rs_sigma_24h_abs': 2.0, 'rs_sigma_4h_abs': 2.5, 'funding_pct_hi': 0.95, 'funding_pct_lo': 0.05, 'oi_24h_abs_pct': 10.0, 'basis_abs_bp': 25.0}): geometry looks at these too
- BTCUSDT: funding p4
- ETHUSDT: funding p4
- SOLUSDT: funding p0
- XRPUSDT: funding p1
- SOXLUSDT: funding p2; basis +59bp
- DOGEUSDT: funding p5
- NVDAUSDT: basis +54bp
- XAGUSDT: OI24h +11.8%
- EVAAUSDT: rs24h +2.0sd
- SOPHUSDT: rs24h -2.3sd; basis -65bp
- INITUSDT: OI24h +19.3%
- WLDUSDT: funding p0
