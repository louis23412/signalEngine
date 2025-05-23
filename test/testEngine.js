import fs from 'fs';
import path from 'path';
import readline from 'readline';
import NeuralSignalEngine from '../src/signals.js';

// Circular buffer implementation
class CircularBuffer {
    constructor(size) {
        this.size = size;
        this.buffer = new Array(size);
        this.head = 0;
        this.length = 0;
    }

    push(item) {
        this.buffer[this.head] = item;
        this.head = (this.head + 1) % this.size;
        if (this.length < this.size) {
            this.length++;
        }
    }

    getArray() {
        if (this.length < this.size) {
            return this.buffer.slice(0, this.length);
        }
        const arr = [];
        for (let i = 0; i < this.size; i++) {
            arr.push(this.buffer[(this.head - this.length + i + this.size) % this.size]);
        }
        return arr;
    }
}

const maxCacheSize = 100; // Optimal cache size
const minCacheSize = 100; // Minimum cache size for processing
const batchSize = 1; // Process each new candle, mimicking real-time
const numRuns = 100000; // Number of backtest runs
const stateDir = path.join(import.meta.dirname, '..', 'state');
const results = []; // Store results for each run

// Function to delete state folder
const clearStateFolder = () => {
    if (fs.existsSync(stateDir)) {
        fs.rmSync(stateDir, { recursive: true, force: true });
    }
};

const renameFolder = (oldName, newName) => {
  try {
    fs.renameSync(oldName, newName);
    console.log(`Folder renamed from ${oldName} to ${newName}`);
  } catch (error) {
    if (error.code === 'ENOENT') {
      console.log(`Error: Folder '${oldName}' not found`);
    } else if (error.code === 'EEXIST') {
      console.log(`Error: A folder named '${newName}' already exists`);
    } else {
      console.log(`Error: ${error.message}`);
    }
  }
}

const calculateScore = (accuracy, winRate, avgReward, trades) => {
  const normalizedAvgReward = avgReward * 10000; // Scale avgReward to comparable range
  const normalizedTrades = (trades / 50000) * 100; // Normalize trades assuming 50,000 max
  return (
    0.2 * accuracy +
    0.25 * winRate +
    0.75 * normalizedAvgReward +
    0.1 * normalizedTrades
  );
};

let bestScore = 55.39435
let currentScore = 0
let signal;

// Function to run a single backtest
async function runBacktest(runNumber) {
    return new Promise((resolve, reject) => {
        console.log(`Starting Run ${runNumber}...`);
        // clearStateFolder(); // Delete state folder before run
        const engine = new NeuralSignalEngine();
        const cache = new CircularBuffer(maxCacheSize);
        let totalCandles = 0;
        let batchBuffer = [];

        const rd = readline.createInterface({
            input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
        });

        rd.on('line', (line) => {
            try {
                const candleObj = JSON.parse(line);
                cache.push(candleObj);
                batchBuffer.push(candleObj);
                totalCandles++;

                if (totalCandles % 1000 === 0) {
                    console.log(`Run ${runNumber} Progress: ${totalCandles} candles`);
                }
                if (totalCandles % 20000 === 0) {
                    console.log(signal)
                    process.exit()
                }

                if (batchBuffer.length >= batchSize && cache.length >= minCacheSize) {
                    try {
                        signal = engine.getSignal(cache.getArray());
                        currentScore = calculateScore(signal.totalAccuracy, signal.performanceWinRate, signal.performanceAvgReward, signal.totalTrades)
                    } catch (e) {
                        console.error(`Error in getSignal (Run ${runNumber}):`, e);
                        reject(e);
                        return;
                    }
                    batchBuffer = []; // Clear batch
                }
            } catch (err) {
                console.error(`Error parsing line (Run ${runNumber}):`, err);
            }
        });

        rd.on('close', () => {
            try {
                console.log(`Run ${runNumber} Completed. Total Candles: ${totalCandles}`);

                // Delete specific files in stateDir, preserving learning_state.json and neural_state.json
                // const filesToDelete = [
                //     'lifetime_state.json',
                //     'performance_summary.json',
                //     'candle_embeddings.json'
                // ];

                // filesToDelete.forEach(file => {
                //     const filePath = path.join(stateDir, file);
                //     try {
                //         if (fs.existsSync(filePath)) {
                //             fs.unlinkSync(filePath);
                //             console.log(`Deleted ${file}`);
                //         }
                //     } catch (err) {
                //         console.error(`Error deleting ${file}: ${err}`);
                //     }
                // });

                console.log(signal)
                process.exit()
                // console.log(currentScore)
                // const newObj = {
                //     currentScore,
                //     signal
                // }
                // fs.appendFileSync('./scores.jsonl', `${JSON.stringify(newObj)}\n`)

                // if (currentScore > bestScore) {
                //     console.log('Best score!:', currentScore)
                //     bestScore = currentScore

                //     renameFolder(stateDir, path.join(import.meta.dirname, '..', `${bestScore}`));

                //     console.log(signal)
                // } else {
                //     console.log(currentScore)
                //     console.log(signal)
                // }

                resolve({
                    run: runNumber,
                    totalAccuracy: signal.totalAccuracy,
                    performanceWinRate: signal.performanceWinRate,
                    performanceAvgReward: signal.performanceAvgReward,
                    totalTrades: signal.totalTrades,
                    totalCandles
                });
            } catch (e) {
                console.error(`Error finalizing Run ${runNumber}:`, e);
                console.log(e)
                process.exit()
                reject(e);
            }
        });

        rd.on('error', (err) => {
            console.error(`Readline error (Run ${runNumber}):`, err);
            reject(err);
        });
    });
}

// Function to run all backtests and generate report
async function runAllBacktests() {
    for (let i = 1; i <= numRuns; i++) {
        try {
            const result = await runBacktest(i);
            results.push(result);
            if (i < numRuns) {
                // console.log(`Waiting 1 minute before starting Run ${i + 1}...`);
                // await new Promise(resolve => setTimeout(resolve, 60000)); // 1-minute delay
            }
        } catch (err) {
            console.error(`Run ${i} failed:`, err);
            console.log(err)
            process.exit()
            results.push({ run: i, totalAccuracy: 0, performanceWinRate: 0, performanceAvgReward: 0, totalTrades: 0, totalCandles: 0 });
            if (i < numRuns) {
                console.log(`Waiting 3 minutes before starting Run ${i + 1}...`);
                await new Promise(resolve => setTimeout(resolve, 180000)); // 3-minute delay even on failure
            }
        }
    }

    // Generate report
    console.log('\n=== Backtest Report ===');
    console.log('Individual Run Results:');
    results.forEach((result) => {
        console.log(
            `Run ${result.run}: Accuracy = ${result.totalAccuracy.toFixed(2)}%, ` +
            `Win Rate = ${result.performanceWinRate.toFixed(2)}%, ` +
            `Avg Reward = ${result.performanceAvgReward.toFixed(8)}, ` +
            `Trades = ${result.totalTrades}, ` +
            `Candles = ${result.totalCandles}`
        );
    });

    // Calculate average and standard deviation
    const validResults = results.filter(r => r.totalAccuracy > 0);
    if (validResults.length === 0) {
        console.log('No valid runs to compute statistics.');
        return;
    }

    const avgAccuracy = validResults.reduce((sum, r) => sum + r.totalAccuracy, 0) / validResults.length;
    const avgWinRate = validResults.reduce((sum, r) => sum + r.performanceWinRate, 0) / validResults.length;
    const avgReward = validResults.reduce((sum, r) => sum + r.performanceAvgReward, 0) / validResults.length;
    const avgTotalTrades = validResults.reduce((sum, r) => sum + r.totalTrades, 0) / validResults.length;
    const stdDevAccuracy = Math.sqrt(
        validResults.reduce((sum, r) => sum + (r.totalAccuracy - avgAccuracy) ** 2, 0) / validResults.length
    );

    console.log('\nAggregate Statistics:');
    console.log(`Number of Valid Runs: ${validResults.length}`);
    console.log(`Average Accuracy: ${avgAccuracy.toFixed(2)}%`);
    console.log(`Standard Deviation (Accuracy): ${stdDevAccuracy.toFixed(2)}%`);
    console.log(`Average Session Win Rate: ${avgWinRate.toFixed(2)}%`);
    console.log(`Average Session Reward: ${avgReward.toFixed(8)}`);
    console.log(`Average Total Trades: ${avgTotalTrades.toFixed(0)}`);
}

// Execute backtests
runAllBacktests().catch((err) => {
    console.error('Error running backtests:', err);
    process.exit(1);
});