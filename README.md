# NeuralSignalEngine

## Overview
NeuralSignalEngine is a JavaScript-based trading signal generator that leverages an ensemble of transformer neural networks combined with technical indicators to produce trading signals. It processes financial market candlestick data to compute indicators, train the model, and generate buy/hold signals with associated confidence levels, target prices, and stop-loss levels. All trades managed internally by the engine are simulated and not executed in real markets.

## Features
- **Transformer Ensemble**: Utilizes multiple transformer models with multi-head attention and feed-forward layers for robust signal prediction.
- **Technical Indicators**: Computes RSI, MACD, ATR, and volume-based metrics to analyze market conditions.
- **Q-Learning**: Employs a Q-table for reinforcement learning to optimize trading decisions.
- **SQLite Database**: Persists transformer parameters, trading patterns, simulated open trades, and candle data.
- **Dynamic Thresholding**: Adjusts confidence thresholds based on market volatility and conditions.

## Requirements
- Node.js (v16 or higher)
- Dependencies:
  - `better-sqlite3` for database operations
  - `fs` and `path` (Node.js built-in modules)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/louis23412/signalEngine
   cd signalEngine
   ```
2. Install dependencies:
   ```bash
   npm install better-sqlite3
   ```
3. Ensure the `state` directory is writable for SQLite database storage (`neural_engine.db`).

## Training the Model
Before trusting the generated signals, the model must be trained with substantial historical candlestick data. The transformer neural network and Q-table learn from closed simulated trades. Provide a dataset of 10,000+ candles covering diverse market conditions and run `getSignal` multiple times to allow the model to learn patterns and adjust weights. Signals may be unreliable until the model has been sufficiently trained.

## Usage
1. **Import and Initialize**:
   ```javascript
   import NeuralSignalEngine from './src/signals.js';

   const engine = new NeuralSignalEngine();
   ```

2. **Prepare Candlestick Data**: Provide an array of candlestick objects with the following structure, using timestamps in milliseconds (e.g., from `Date.now()`). Ensure all fields are valid numbers and `volume` is non-negative:
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

## Configuration Options
The engine includes configurable parameters in the `NeuralSignalEngine` class:
- `minMultiplier`: Minimum multiplier for trade sizing (default: 1).
- `maxMultiplier`: Maximum multiplier for trade sizing (default: 2.5).
- `baseConfidenceThreshold`: Base threshold for signal confidence (default: 60).
- `atrFactor`: Multiplier for ATR to set target prices (default: 10).
- `stopFactor`: Multiplier for ATR to set stop-loss levels (default: 2.5).
- `learningRate`: Learning rate for Q-table updates (default: 0.25).

These can be modified by updating the `#config` object in the `NeuralSignalEngine` constructor, though changes require careful testing to ensure stability.

## Project Structure
- **signals.js**: Core implementation containing:
  - `Transformer`: Neural network with multi-head attention and feed-forward layers, using an ensemble of three models.
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

## Performance Considerations
- The transformer model is computationally intensive due to its multi-head attention and ensemble approach. Ensure sufficient memory and CPU resources, especially when processing large datasets (10,000+ candles).
- Database operations may slow down with large datasets. Regularly clean up old candles (the engine retains the most recent 1,000 by default).
- Training with 10,000+ candles is recommended to capture diverse market patterns, but this increases initial setup time.

## Troubleshooting
- **Invalid Candle Data**: Ensure all candlestick fields (`timestamp`, `open`, `high`, `low`, `close`, `volume`) are valid numbers, with `volume` non-negative and `timestamp` in milliseconds.
- **Database Errors**: Verify that the `state` directory is writable and not corrupted. Delete `neural_engine.db` to reset the database if needed.
- **Unreliable Signals**: If signals seem inconsistent, ensure the model has been trained with 10,000+ candles across varied market conditions.
- **Performance Issues**: Reduce the number of candles processed or optimize the database by removing old records if performance degrades.

## Notes
- The engine requires at least 11 candles for indicator computation.
- The transformer model is trained incrementally as simulated trades are closed.
- Ensure sufficient disk space for the SQLite database in the `state` directory.
- The system is designed for financial market data but can be adapted for other time-series data with appropriate modifications.

## Disclaimer
NeuralSignalEngine is provided for educational and research purposes only. It generates simulated trading signals and does not execute real trades. The signals are based on historical data and machine learning models, which may not accurately predict future market behavior. Use at your own risk, and do not rely on this tool for actual financial decisions without consulting a qualified financial advisor. The authors and contributors are not liable for any financial losses or damages resulting from the use of this software.

## License
MIT License. See [LICENSE](LICENSE) for details.