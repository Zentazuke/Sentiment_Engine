"""CLI tester for the standalone sentiment engine.

The CLI talks to the running FastAPI service through HTTP so state persists in
that service process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from sentiment_engine.ingestion.mock_social import SCENARIOS, generate_messages

DEFAULT_URL = "http://127.0.0.1:8787"


def _post(url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - local dev tool
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _get(url: str, path: str) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}{path}", timeout=5) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _print(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def cmd_health(args: argparse.Namespace) -> None:
    _print(_get(args.url, "/health"))


def cmd_inject_social(args: argparse.Namespace) -> None:
    _print(
        _post(
            args.url,
            "/ingest/social",
            {
                "symbol": args.symbol,
                "source": args.source,
                "text": args.text,
                "author": args.author,
                "timestamp": time.time(),
            },
        )
    )


def cmd_inject_price(args: argparse.Namespace) -> None:
    _print(
        _post(
            args.url,
            "/ingest/price",
            {
                "symbol": args.symbol,
                "price": args.price,
                "timestamp": time.time(),
            },
        )
    )


def cmd_snapshot(args: argparse.Namespace) -> None:
    symbol_key = args.symbol.replace("/", "-")
    _print(_get(args.url, f"/snapshot/{symbol_key}"))


def cmd_evaluate(args: argparse.Namespace) -> None:
    _print(
        _post(
            args.url,
            "/evaluate",
            {
                "symbol": args.symbol,
                "direction": args.direction,
                "bot_confidence": args.bot_confidence,
                "trigger_price": args.trigger_price,
                "timestamp": time.time(),
            },
        )
    )


def cmd_simulate(args: argparse.Namespace) -> None:
    # Seed a previous quiet/mixed baseline if requested, then inject the active scenario.
    if args.baseline:
        for msg in generate_messages(
            scenario="mixed_chop",
            symbol=args.symbol,
            count=max(1, args.count // 2),
            seconds=110,
        ):
            _post(args.url, "/ingest/social", msg.__dict__)

    start_price = args.price
    for step in range(args.price_steps):
        price = start_price * (1 + (args.price_change_pct / 100) * (step / max(1, args.price_steps - 1)))
        _post(args.url, "/ingest/price", {"symbol": args.symbol, "price": price, "timestamp": time.time()})
        time.sleep(args.delay)

    for msg in generate_messages(scenario=args.scenario, symbol=args.symbol, count=args.count, seconds=args.seconds):
        _post(args.url, "/ingest/social", msg.__dict__)
        if args.delay:
            time.sleep(args.delay)

    print(f"Injected scenario={args.scenario!r} symbol={args.symbol} count={args.count}")
    cmd_snapshot(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test CLI for the standalone sentiment engine")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of running sentiment API")
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health")
    health.set_defaults(func=cmd_health)

    inject_social = sub.add_parser("inject-social")
    inject_social.add_argument("--symbol", required=True)
    inject_social.add_argument("--text", required=True)
    inject_social.add_argument("--source", default="manual")
    inject_social.add_argument("--author", default=None)
    inject_social.set_defaults(func=cmd_inject_social)

    inject_price = sub.add_parser("inject-price")
    inject_price.add_argument("--symbol", required=True)
    inject_price.add_argument("--price", required=True, type=float)
    inject_price.set_defaults(func=cmd_inject_price)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--symbol", required=True)
    snapshot.set_defaults(func=cmd_snapshot)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--symbol", required=True)
    evaluate.add_argument("--direction", choices=["STRAT_LONG", "STRAT_SHORT"], required=True)
    evaluate.add_argument("--bot-confidence", type=float, default=0.70)
    evaluate.add_argument("--trigger-price", type=float, default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    simulate = sub.add_parser("simulate")
    simulate.add_argument("--symbol", required=True)
    simulate.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    simulate.add_argument("--count", type=int, default=20)
    simulate.add_argument("--seconds", type=int, default=30)
    simulate.add_argument("--baseline", action="store_true", help="Inject a mixed baseline before the scenario")
    simulate.add_argument("--price", type=float, default=100000.0)
    simulate.add_argument("--price-change-pct", type=float, default=0.25)
    simulate.add_argument("--price-steps", type=int, default=4)
    simulate.add_argument("--delay", type=float, default=0.0)
    simulate.set_defaults(func=cmd_simulate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI display
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
