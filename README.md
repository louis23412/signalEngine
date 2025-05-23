# NeuralSignalEngine

## Overview
NeuralSignalEngine is a JavaScript-based trading signal generator that uses a transformer neural network architecture combined with technical indicators to produce trading signals. It processes financial market data (candlestick data) to compute indicators, train a transformer model, and generate buy/hold signals with associated confidence levels, target prices, and stop-loss levels. All trades managed internally by the engine are simulated and not executed in real markets.

## Features
- **Transformer Neural Network**: Implements an ensemble of transformer models for signal prediction.
- **Technical Indicators**: Calculates RSI, MACD, ATR, and volume-based metrics to analyze market conditions.
- **Q-Learning**: Uses a Q-table for reinforcement learning to optimize trading decisions.
- **SQLite Database**: Persists transformer parameters, trading patterns, open trades, and candle data.
- **Dynamic Thresholding**: Adjusts confidence thresholds based on market volatility and conditions.

## Requirements
- Node.js (v16 or higher)
- Dependencies:
  - `better-sqlite3` for database operations
  - `fs` and `path` (Node.js built-in modules)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```bash
   npm install better-sqlite3
   ```
3. Ensure the `state` directory is writable for SQLite database storage.

## Training the Model
Before trusting the generated signals, the model must be trained with sufficient historical candlestick data. The transformer neural network and Q-table learn from closed trades, which are simulated within the engine. Provide a large dataset (e.g., 1000+ candles) and run `getSignal` multiple times to allow the model to learn patterns and adjust weights. Signals may be unreliable until the model has been trained with diverse market conditions.

## Usage
1. **Import and Initialize**:
   ```javascript
   import NeuralSignalEngine from './signals.js';

   const engine = new NeuralSignalEngine();
   ```

2. **Prepare Candlestick Data**:
   Provide an array of candlestick objects with the following structure, using timestamps in milliseconds (e.g., from `Date.now()`):
   ```javascript
   const candles = [
     {
       timestamp: 1672531200000, // Milliseconds (e.g., 2023-01-01T00:00:00Z)
       open: 100,
       high: 105,
       low: 95,
       close: 102,
       volume: 1000
     },
     // Additional candles...
   ];
   ```

3. **Generate Signal**:
   ```javascript
   const signal = engine.getSignal(candles);
   console.log(signal);
   ```
   Example output:
   ```javascript
   {
     currentConfidence: 75.123,
     suggestedConfidence: 60.456,
     multiplier: 1.789,
     sellPrice: 105.23,
     stopLoss: 98.45,
     expectedReward: 0.029,
     suggestedAction: 'buy'
   }
   ```

## Project Structure
- **signals.js**: Core implementation containing:
  - `Transformer`: Neural network with multi-head attention and feed-forward layers.
  - `IndicatorProcessor`: Computes technical indicators (RSI, MACD, ATR).
  - `NeuralSignalEngine`: Main class integrating transformer, indicators, and database.
- **state/**: Directory for SQLite database storage (`neural_engine.db`).

## Database Schema
The engine uses a SQLite database to store:
- **qtable**: Q-learning values for buy/hold actions.
- **patterns**: Historical trading patterns and their scores.
- **open_trades**: Simulated active trades with entry prices, sell prices, and stop-loss levels.
- **candles**: Historical candlestick data with timestamps in milliseconds.
- **transformer_parameters**: Transformer model parameters and ensemble weights.

## Notes
- The engine assumes valid candlestick data with at least 11 candles for indicator computation.
- The transformer model is trained incrementally as simulated trades are closed.
- Ensure sufficient disk space for the SQLite database in the `state` directory.
- The system is designed for financial market data but can be adapted for other time-series data.

## Disclaimer
NeuralSignalEngine is provided for educational and research purposes only. It generates simulated trading signals and does not execute real trades. The signals are based on historical data and machine learning models, which may not accurately predict future market behavior. Use at your own risk, and do not rely on this tool for actual financial decisions without consulting a qualified financial advisor. The authors and contributors are not liable for any financial losses or damages resulting from the use of this software.

## License
MIT License. See [LICENSE](LICENSE) for details.