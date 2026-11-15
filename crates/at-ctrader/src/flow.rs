//! Multi-step flows shared by `atx ctrader` and the tests: connection check and account login.

use serde::Serialize;

use crate::creds::CtraderCreds;
use crate::pb;
use crate::session::{self, Session};

#[derive(Clone, Debug, Serialize)]
pub struct AccountRow {
    pub ctid_trader_account_id: u64,
    pub is_live: bool,
    pub broker: String,
    pub trader_login: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CheckReport {
    pub env: String,
    pub server_version: String,
    pub scope: String,
    pub read_only: bool,
    pub accounts: Vec<AccountRow>,
    pub account: Option<AccountRow>,
    pub broker_name: Option<String>,
    pub deposit_asset: Option<String>,
    pub account_type: Option<String>,
    pub swap_free: Option<bool>,
    pub leverage: Option<f64>,
    pub n_symbols: Option<usize>,
    pub n_categories: Option<usize>,
}

fn row(a: &pb::ProtoOaCtidTraderAccount) -> AccountRow {
    AccountRow {
        ctid_trader_account_id: a.ctid_trader_account_id,
        is_live: a.is_live.unwrap_or(false),
        broker: a.broker_title_short.clone().unwrap_or_default(),
        trader_login: a.trader_login,
    }
}

/// Application auth -> account list (scope check) -> account auth. Returns the ctid account id.
/// Fails before account auth if the token can trade.
pub async fn login(sess: &mut Session, c: &CtraderCreds) -> anyhow::Result<(i64, pb::ProtoOaGetAccountListByAccessTokenRes)> {
    sess.app_auth(c).await?;
    let accts = sess.accounts(c).await?;
    session::ensure_read_only(accts.permission_scope)?;
    let a = session::pick_account(&accts.ctid_trader_account, c.account_id, c.env)?;
    let id = a.ctid_trader_account_id as i64;
    sess.account_auth(id, c).await?;
    Ok((id, accts))
}

/// Everything `atx ctrader check` prints. On a trading-scope token it returns the partial report and the
/// refusal error (the caller prints both and exits non-zero).
pub async fn check(sess: &mut Session, c: &CtraderCreds) -> (CheckReport, Option<anyhow::Error>) {
    let mut r = CheckReport {
        env: c.env.as_str().into(),
        server_version: String::new(),
        scope: String::new(),
        read_only: false,
        accounts: vec![],
        account: None,
        broker_name: None,
        deposit_asset: None,
        account_type: None,
        swap_free: None,
        leverage: None,
        n_symbols: None,
        n_categories: None,
    };
    let res: anyhow::Result<()> = async {
        sess.app_auth(c).await.map_err(|e| anyhow::anyhow!("application auth failed (check CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET): {e}"))?;
        r.server_version = sess.version().await.unwrap_or_default();
        let accts = sess.accounts(c).await.map_err(|e| anyhow::anyhow!("account list by access token failed (expired or wrong CTRADER_ACCESS_TOKEN?): {e}"))?;
        r.scope = session::scope_name(accts.permission_scope).into();
        r.accounts = accts.ctid_trader_account.iter().map(row).collect();
        session::ensure_read_only(accts.permission_scope)?;
        r.read_only = true;
        let a = session::pick_account(&accts.ctid_trader_account, c.account_id, c.env)?;
        r.account = Some(row(&a));
        let id = a.ctid_trader_account_id as i64;
        sess.account_auth(id, c).await?;
        let t = sess.trader(id).await?;
        let assets = sess.assets(id).await.unwrap_or_default();
        r.broker_name = t.broker_name.clone();
        r.deposit_asset = assets.iter().find(|x| x.asset_id == t.deposit_asset_id).map(|x| x.name.clone()).or(Some(format!("asset id {}", t.deposit_asset_id)));
        r.account_type = t.account_type.and_then(|v| pb::ProtoOaAccountType::try_from(v).ok()).map(|v| v.as_str_name().to_string());
        r.swap_free = t.swap_free;
        r.leverage = t.leverage_in_cents.map(|l| l as f64 / 100.0);
        r.n_symbols = Some(sess.symbols(id).await?.len());
        r.n_categories = sess.categories(id).await.ok().map(|v| v.len());
        Ok(())
    }
    .await;
    (r, res.err())
}
