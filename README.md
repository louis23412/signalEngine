# NeuralSignalEngine

## Overview
NeuralSignalEngine is a JavaScript-based tool for generating simulated trading signals using an ensemble of transformer neural networks and technical indicators. It processes financial market candlestick data to compute indicators, train its model, and produce buy/hold signals with confidence levels, target prices, and stop-loss levels. All trades are simulated and not executed in real markets.

## Features
- **Transformer Ensemble**: Uses five transformer models with multi-head attention for robust signal prediction.
- **Technical Indicators**: Calculates RSI, MACD, ATR, and volume metrics for market analysis.
- **Q-Learning**: Optimizes trading decisions using a Q-table based on historical patterns.
- **SQLite Database**: Stores transformer parameters, trading patterns, simulated trades, and candlestick data for persistence across sessions.
- **Dynamic Thresholding**: Adapts confidence thresholds based on market volatility.

## Requirements
- **Node.js**: Version 16 or higher.
- **Dependencies**: Install `better-sqlite3` via `npm install better-sqlite3`.
- **Writable Directory**: Ensure the `state` directory (auto-created in project root) is writable for the SQLite database (`neural_engine.db`).

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

## Project Structure
- **`src/signals.js`**: Core logic with `Transformer`, `IndicatorProcessor`, and `NeuralSignalEngine` classes.
- **`test/`**:
  - `testEngine.js`: Script for training and testing the engine.
  - `candles.jsonl`: Input file with candlestick data (must be created).
- **`state/`**: Directory for SQLite database (`neural_engine.db`).
- **`README.md`**: Project documentation.

## Training and Testing
The `testEngine.js` script trains the model and tests signal generation using candlestick data from `test/candles.jsonl`. Training requires at least 10,000 candles covering varied market conditions for reliable signals. The model learns incrementally as it processes data and updates its transformer weights and Q-table.

The script maintains a sliding window of the latest 100 candles, mimicking real-world usage where the engine is called with each new candle received to generate a signal. This fixed window ensures efficient processing by focusing on recent market data.

### Steps
1. **Prepare `test/candles.jsonl`**:
   Create a `candles.jsonl` file in the `test/` folder, with each line as a JSON object representing a candlestick. Example format:
   ```json
   {"timestamp":1672531200000,"open":100.0,"high":105.0,"low":95.0,"close":102.0,"volume":1000}
   ```
   - **Fields**: `timestamp` (milliseconds, e.g., from `Date.now()`), `open`, `high`, `low`, `close` (numbers), `volume` (non-negative number).
   - **Tip**: Validate the JSONL file to ensure all fields are numeric and correctly formatted to avoid processing errors.
2. **Run `testEngine.js`**:
   ```bash
   node test/testEngine.js
   ```
   - The script processes `test/candles.jsonl`, updates the sliding window with each new candle, and generates signals.
   - It displays progress, processing time, and ETA, and outputs the final signal.
   - Example output:
     ```javascript
     {
       currentConfidence: 75.123, // Confidence in the signal (0-100)
       suggestedConfidence: 60.456, // Dynamic threshold
       multiplier: 1.789, // Trade sizing multiplier
       sellPrice: 105.23, // Target sell price
       stopLoss: 98.45, // Stop-loss price
       expectedReward: 0.029, // Expected reward ratio
       suggestedAction: 'buy' // 'buy' or 'hold'
     }
     ```

3. **Training Process**:
   - The engine trains incrementally as it processes candles, updating transformer weights and the Q-table based on simulated trade outcomes.
   - Use large datasets (10,000+ candles) for better signal accuracy.
   - **Note**: At least 11 candles are required to compute technical indicators (e.g., RSI, MACD, ATR).

## Usage
To use the engine programmatically:
1. **Import and Initialize**:
   ```javascript
   import NeuralSignalEngine from './src/signals.js';
   const engine = new NeuralSignalEngine();
   ```
2. **Provide Candlestick Data**:
   ```javascript
   const candles = [
     { timestamp: 1672531200000, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000 },
     // More candles...
   ];
   ```
3. **Generate Signal**:
   ```javascript
   const signal = engine.getSignal(candles);
   console.log(signal);
   ```

## Configuration
Adjust parameters in the `#config` object within the `NeuralSignalEngine` class in `src/signals.js`:
- `minMultiplier`: Minimum trade sizing (default: 1).
- `maxMultiplier`: Maximum trade sizing (default: 2.5).
- `baseConfidenceThreshold`: Base confidence threshold (default: 60).
- `atrFactor`: ATR multiplier for target prices (default: 10).
- `stopFactor`: ATR multiplier for stop-loss (default: 2.5).
- `learningRate`: Q-table learning rate (default: 0.25).

Test thoroughly after modifying these settings, as changes can significantly impact signal generation.

## Database Schema
The SQLite database (`state/neural_engine.db`) ensures persistence of learned parameters and includes:
- **`qtable`**: Q-learning values (`state_key`, `buy`, `hold`).
- **`patterns`**: Trading patterns (`bucket_key`, `features`, `score`).
- **`open_trades`**: Simulated trades (`timestamp`, `sellPrice`, `stopLoss`, `entryPrice`, `confidence`, `candlesHeld`, `strategy`, `patternScore`, `features`, `stateKey`, `dynamicThreshold`).
- **`candles`**: Historical candles (`timestamp`, `open`, `high`, `low`, `close`, `volume`).
- **`transformer_parameters`**: Transformer weights (`transformer_id`, `parameters`, `ensemble_weight`, `updated_at`).

## Performance Tips
- **Data Size**: Use 10,000+ candles for robust training to improve signal accuracy.
- **Database Cleanup**: The engine retains the 1,000 most recent candles. Clear old data if the database grows large.
- **Input Validation**: Ensure `test/candles.jsonl` contains valid JSONL with numeric fields to prevent runtime errors.
- **Sliding Window Adjustment**: The sliding window of 100 candles can be adjusted by modifying `cacheSize` in `testEngine.js`, but test thoroughly to balance memory and performance.

## Troubleshooting
- **Invalid `candles.jsonl`**: Ensure each line is a valid JSON object with correct fields and numeric values. Check for missing or invalid `timestamp`, `volume` (must be non-negative), or other fields.
- **Database Issues**: If `neural_engine.db` is corrupted, delete it to reset (this clears learned parameters).
- **Unreliable Signals**: Train with diverse, large datasets (10,000+ candles).
- **Performance Slowdowns**: Monitor `testEngine.js` output for processing times and reduce candle volume or adjust `cacheSize` if needed.

## Notes
- Requires at least 11 candles for indicator calculations.
- Signals improve with more training data.
- Ensure sufficient disk space for `neural_engine.db`.
- The SQLite database persists transformer weights and Q-table, allowing the model to resume learning across sessions.

## Disclaimer
This tool is for educational and research purposes only. It generates simulated signals, not real trades. Signals may not predict future market performance. Use at your own risk and consult a financial advisor before making decisions. The author is not liable for any losses.

## License
MIT License. See [LICENSE](LICENSE) for details.