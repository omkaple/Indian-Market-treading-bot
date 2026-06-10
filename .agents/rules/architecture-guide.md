# Architecture Guide

Adhere to the following architectural design principles:

## 1. Decoupled Ingestion-Consumer Pattern
- The live WebSocket tick ingestion layer (Producer) must run on a separate background execution thread from the user dashboard and strategy calculations.
- Use a thread-safe `queue.Queue` to share ticks from the Producer thread to the processing consumer thread.
- This decoupling eliminates data blockages and prevents tick stream drops/lag during high volatility trading windows.

## 2. Dynamic Database Segregation
- Each stock asset (e.g. CANBK, SBIN) must save its data into an isolated database collection named:
  `f"{selected_stock.lower()}_5min_candles"`
- Do not mix data of different stock instruments in the same collection.

## 3. Thread Safe Locks
- Use `threading.Lock` when modifying shared objects (such as the active candle buffers) to prevent race conditions during tick updates.
