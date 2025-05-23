import fs from 'fs';
import path from 'path';
import readline from 'readline';
import NeuralSignalEngine from '../src/signals.js';

const engine = new NeuralSignalEngine();

const cacheSize = 100; // Fixed cache size
const cache = []; // Simple array to store candles

let signal;
let totalCandles = 0;

const rd = readline.createInterface({
    input: fs.createReadStream(path.join(import.meta.dirname, 'candles.jsonl'))
});

rd.on('line', (line) => {
    const candleObj = JSON.parse(line);
    cache.push(candleObj);
    if (cache.length > cacheSize) {
        cache.shift(); // Remove oldest candle to maintain cache size
    }
    totalCandles++;

    console.log(`Progress: ${totalCandles} candles`);

    if (cache.length >= cacheSize) {
        try {
            signal = engine.getSignal(cache.slice(-cacheSize)); // Use last 100 candles
        } catch (e) {
            console.log(e);
            process.exit();
        }
    }
});

rd.on('close', () => {
    console.log(`Completed. Total Candles: ${totalCandles}`);
    console.log(signal);
});