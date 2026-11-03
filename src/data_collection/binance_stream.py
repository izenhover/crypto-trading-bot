"""دریافت داده لحظه‌ای از Binance WebSocket"""
import json
import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional
import websockets
from loguru import logger


BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_WS_TESTNET = "wss://stream.testnet.binance.vision/ws"


class BinanceStreamer:
    """جریان داده لحظه‌ای Binance"""

    def __init__(self, symbols: list, intervals: list, use_testnet: bool = False):
        self.symbols = [s.lower() for s in symbols]
        self.intervals = intervals
        self.base_url = BINANCE_WS_TESTNET if use_testnet else BINANCE_WS_BASE
        self.callback: Optional[Callable] = None
        self._running = False

    def _build_stream_names(self):
        names = []
        for sym in self.symbols:
            for interval in self.intervals:
                names.append(f"{sym}@kline_{interval}")
        return names

    async def _listen_single(self, stream_name: str):
        url = f"{self.base_url}/{stream_name}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info(f"Connected: {stream_name}")
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if self.callback:
                            self.callback(data)
            except Exception as e:
                logger.error(f"Stream error {stream_name}: {e}")
                await asyncio.sleep(5)

    async def _run_all(self):
        tasks = [self._listen_single(name) for name in self._build_stream_names()]
        await asyncio.gather(*tasks)

    def start(self, callback: Callable, duration_seconds: Optional[int] = None):
        """شروع جریان داده. اگر duration بدهید، بعد از آن متوقف می‌شود."""
        self.callback = callback
        self._running = True

        async def _main():
            if duration_seconds:
                try:
                    await asyncio.wait_for(self._run_all(), timeout=duration_seconds)
                except asyncio.TimeoutError:
                    logger.info(f"Stream stopped after {duration_seconds}s")
            else:
                await self._run_all()

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user")
        finally:
            self._running = False
