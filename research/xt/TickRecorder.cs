// XT TickRecorder: a cTrader Automate cBot that records TRUE ticks (every bid / ask change) of a list of symbols to CSV.
// Agent XT, 2026-11-28. Data only: this cBot never opens, modifies or closes a position or an order, and never reads
// account fields (balance, equity, margin). It only subscribes to the symbols' ticks and appends them to files.
//
// One file per symbol and UTC day: <Output folder>\<SYMBOL>\<yyyy-MM-dd>.csv, header
//     tick_ms,event_server_ms,local_ms,bid,ask,spread
//   tick_ms          the tick's own server timestamp (last tick of MarketData.GetTicks), epoch ms UTC
//   event_server_ms  Server.TimeInUtc when the Tick event ran, epoch ms UTC
//   local_ms         this PC's clock (DateTime.UtcNow), epoch ms UTC: the same clock as the Python recorders
//                    (scripts/xt_bybit_l1.py recv_ts), so the two venues can be aligned without a clock model
//   bid, ask, spread price units (spread = ask - bid), invariant culture
// SPY symbols are skipped (the user's standing rule). Files are flushed every FlushSeconds and closed on stop.
//
// Build and run (cTrader Desktop):
//   1. Algo tab (Automate) -> cBots -> New -> name it TickRecorder; replace the template with this file; Build (Ctrl+B).
//   2. Add an instance on any chart (e.g. BTCUSD, m1). Parameters: Symbols (comma-separated names as shown in cTrader),
//      Output folder (e.g. C:\Users\Asus\Desktop\AI_Trader\data\raw\XT\ctrader_ticks), FlushSeconds (default 5).
//   3. Run it LOCALLY (not as a cloud instance: a cloud instance cannot write to this PC). Accept the file-system access
//      request on the first start.
//   4. Leave cTrader open. Stop the instance to close the files; restart appends to the same day's files.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FileSystem)]
    public class TickRecorder : Robot
    {
        [Parameter("Symbols", DefaultValue = "BTCUSD,ETHUSD,XAUUSD,XAGUSD,Copper,NAS100,USTEC.F,QQQ,QQQ-24,BRENT,UKBRENT.F,WTI,USOIL.F,US500,SPX500,US500.F")]
        public string SymbolList { get; set; }

        [Parameter("Output folder", DefaultValue = "C:\\cTraderTicks")]
        public string Folder { get; set; }

        [Parameter("Flush every (s)", DefaultValue = 5, MinValue = 1, MaxValue = 300)]
        public int FlushSeconds { get; set; }

        private static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        private readonly Dictionary<string, StreamWriter> _writers = new Dictionary<string, StreamWriter>();
        private readonly Dictionary<string, string> _writerDay = new Dictionary<string, string>();
        private readonly Dictionary<string, Ticks> _ticks = new Dictionary<string, Ticks>();
        private readonly Dictionary<string, long> _count = new Dictionary<string, long>();
        private readonly List<Symbol> _symbols = new List<Symbol>();

        protected override void OnStart()
        {
            Directory.CreateDirectory(Folder);
            foreach (var raw in SymbolList.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries))
            {
                var name = raw.Trim();
                if (name.ToUpperInvariant().StartsWith("SPY"))
                {
                    Print("skipped {0}: SPY symbols are excluded", name);
                    continue;
                }
                var sym = Symbols.GetSymbol(name);
                if (sym == null)
                {
                    Print("symbol not found: {0}", name);
                    continue;
                }
                _symbols.Add(sym);
                _ticks[sym.Name] = MarketData.GetTicks(sym.Name);
                _count[sym.Name] = 0;
                sym.Tick += OnSymbolTick;
            }
            Print("recording {0} symbols to {1}", _symbols.Count, Folder);
            Timer.Start(FlushSeconds);
        }

        private static long Ms(DateTime t)
        {
            return (long)(t.ToUniversalTime() - Epoch).TotalMilliseconds;
        }

        private StreamWriter Writer(string name, DateTime utc)
        {
            var day = utc.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
            StreamWriter w;
            if (_writers.TryGetValue(name, out w) && _writerDay[name] == day)
                return w;
            if (w != null)
                w.Dispose();
            var dir = Path.Combine(Folder, name.Replace('/', '_'));
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, day + ".csv");
            var isNew = !File.Exists(path);
            w = new StreamWriter(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read));
            if (isNew)
                w.WriteLine("tick_ms,event_server_ms,local_ms,bid,ask,spread");
            _writers[name] = w;
            _writerDay[name] = day;
            return w;
        }

        private void OnSymbolTick(SymbolTickEventArgs e)
        {
            var local = DateTime.UtcNow;
            var server = Server.TimeInUtc;
            var s = e.Symbol;
            long tickMs = Ms(server);
            Ticks t;
            if (_ticks.TryGetValue(s.Name, out t) && t.Count > 0)
                tickMs = Ms(t.LastTick.Time);
            var w = Writer(s.Name, server);
            w.Write(tickMs.ToString(CultureInfo.InvariantCulture));
            w.Write(',');
            w.Write(Ms(server).ToString(CultureInfo.InvariantCulture));
            w.Write(',');
            w.Write(Ms(local).ToString(CultureInfo.InvariantCulture));
            w.Write(',');
            w.Write(e.Bid.ToString("R", CultureInfo.InvariantCulture));
            w.Write(',');
            w.Write(e.Ask.ToString("R", CultureInfo.InvariantCulture));
            w.Write(',');
            w.WriteLine((e.Ask - e.Bid).ToString("R", CultureInfo.InvariantCulture));
            _count[s.Name] += 1;
        }

        protected override void OnTimer()
        {
            foreach (var w in _writers.Values)
                w.Flush();
        }

        protected override void OnStop()
        {
            foreach (var w in _writers.Values)
            {
                w.Flush();
                w.Dispose();
            }
            _writers.Clear();
            foreach (var kv in _count)
                Print("{0}: {1} ticks recorded", kv.Key, kv.Value);
        }
    }
}
