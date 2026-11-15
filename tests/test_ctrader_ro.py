import pytest

from ai_trader.ctrader import mcp_ro


def _client():
    c = object.__new__(mcp_ro.CTraderRO)
    c.sid, c._id = None, 0
    c._post = lambda body: pytest.fail("no request may be sent")  # noqa: ARG005
    return c


@pytest.mark.parametrize("tool", ["place_market_order", "place_limit_order", "close_all_positions", "amend_position",
                                  "get_balance", "get_accounts_list", "startPlugin", "cancel_all_pending_orders"])
def test_refuses_non_readonly_tools(tool):
    with pytest.raises(PermissionError):
        _client().call(tool)


def test_refuses_spy():
    with pytest.raises(PermissionError):
        _client().call("get_spot_prices", symbolName="SPY")


def test_allowlist_has_no_trading_or_account_tools():
    bad = ("place", "close", "cancel", "amend", "balance", "account", "plugin", "workspace", "chart")
    assert not [t for t in mcp_ro.READ_ONLY_TOOLS if any(b in t.lower() for b in bad)]
