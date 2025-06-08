import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';

const directoryPath = path.join(import.meta.dirname, '..', 'state')

const isValidNumber = (value) => {
  if (value == null) return false;
  const num = typeof value === 'string' ? Number(value) : value;
  return typeof num === 'number' && !isNaN(num) && isFinite(num);
};

class HiveMind {
    #inputSize = 6;
    #hiddenSize = 16;
    #outputSize = 1;
    #numHeads = 4;
    #numLayers = 2;
    #feedForwardSize = 32;
    #ensembleSize = 128;
    #contextWindow = 50;
    #dropoutRate = 0.15;
    #learningRate = 0.01;
    #weightSharingRate = 0.1;
    #diversityWeight = 0.5;
    #attentionScalingFactor = 1.0;
    #gradientClippingThreshold = 1.0;
    #swarmIntelligenceFactor = 0.3;
    #maxPerformanceHistory = 200;
    #maxTrustHistory = 80;
    #adaptiveLearningRate = [];
    #transformers = [];
    #ensembleWeights = [];
    #momentumWeights = [];
    #gradientAccumulation = [];
    #attentionWeightMatrix = [];
    #attentionBias = [];
    #specializationWeights = [];
    #attentionMemory = [];
    #performanceScores = [];
    #agreementScores = [];
    #historicalPerformance = [];
    #trustScoresHistory = [];
    #specializationScores = [];
    #trainingStepCount = 0;

    constructor() {
        this.#performanceScores = Array(this.#ensembleSize).fill(0);
        this.#agreementScores = Array(this.#ensembleSize).fill(0);
        this.#specializationScores = Array(this.#ensembleSize).fill(0);
        this.#ensembleWeights = Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
        this.#historicalPerformance = Array(this.#ensembleSize).fill().map(() => [0]);
        this.#trustScoresHistory = Array(this.#ensembleSize).fill().map(() => [0]);
        this.#adaptiveLearningRate = Array(this.#ensembleSize).fill(this.#learningRate);

        this.#momentumWeights = this.#createWeightStructure();
        this.#gradientAccumulation = this.#createWeightStructure();
        this.#attentionMemory = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#contextWindow).fill().map(() => Array(this.#hiddenSize).fill(0))
        );
        this.#attentionWeightMatrix = Array(this.#ensembleSize).fill().map(() =>
            this.#xavierInit(this.#hiddenSize, 1).map(row => row[0])
        );
        this.#attentionBias = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#hiddenSize).fill().map(() => (Math.random() - 0.5) * Math.sqrt(6 / this.#hiddenSize))
        );
        this.#specializationWeights = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#hiddenSize).fill().map((_, j) =>
                Array(this.#hiddenSize).fill().map((_, k) => {
                    const scale = Math.sqrt(2 / (this.#hiddenSize + this.#hiddenSize)) * (1 + 0.05 * (j + k / this.#hiddenSize));
                    return (Math.random() - 0.5) * scale;
                })
            )
        );

        this.#transformers = Array(this.#ensembleSize).fill().map(() => ({
            positionalEncoding: this.#createPositionalEncoding(),
            attentionWeights: Array(this.#numLayers).fill().map(() => ({
                Wq: this.#xavierInit(this.#hiddenSize, this.#hiddenSize),
                Wk: this.#xavierInit(this.#hiddenSize, this.#hiddenSize),
                Wv: this.#xavierInit(this.#hiddenSize, this.#hiddenSize),
                Wo: this.#xavierInit(this.#hiddenSize, this.#hiddenSize)
            })),
            ffnWeights: Array(this.#numLayers).fill().map(() => ({
                W1: this.#xavierInit(this.#hiddenSize, this.#feedForwardSize),
                W2: this.#xavierInit(this.#feedForwardSize, this.#hiddenSize),
                b1: Array(this.#feedForwardSize).fill(0),
                b2: Array(this.#hiddenSize).fill(0)
            })),
            layerNormWeights: Array(this.#numLayers).fill().map(() => ({
                gamma1: Array(this.#hiddenSize).fill(1),
                beta1: Array(this.#hiddenSize).fill(0),
                gamma2: Array(this.#hiddenSize).fill(1),
                beta2: Array(this.#hiddenSize).fill(0)
            })),
            outputWeights: this.#xavierInit(this.#hiddenSize, this.#outputSize),
            outputBias: Array(this.#outputSize).fill(0)
        }));

        this.#normalizeEnsembleWeights();
        this.#loadState();
    }

    #saveState() {
        const dbPath = path.join(directoryPath, 'hivemind_state.db');
        let db;

        try {
            if (!fs.existsSync(directoryPath)) {
                fs.mkdirSync(directoryPath, { recursive: true });
            }

            db = new Database(dbPath, { fileMustExist: false });
            db.pragma('journal_mode = WAL');
            db.pragma('synchronous = NORMAL');

            db.exec('BEGIN TRANSACTION');

            db.exec(`
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE TABLE IF NOT EXISTS ensemble_weights (
                    idx INTEGER PRIMARY KEY,
                    weight REAL
                );
                CREATE TABLE IF NOT EXISTS performance_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS agreement_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS specialization_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS historical_performance (
                    idx INTEGER,
                    step INTEGER,
                    score REAL,
                    PRIMARY KEY (idx, step)
                );
                CREATE TABLE IF NOT EXISTS trust_scores_history (
                    idx INTEGER,
                    step INTEGER,
                    score REAL,
                    PRIMARY KEY (idx, step)
                );
                CREATE TABLE IF NOT EXISTS adaptive_learning_rate (
                    idx INTEGER PRIMARY KEY,
                    rate REAL
                );
                CREATE TABLE IF NOT EXISTS attention_weight_matrix (
                    idx INTEGER,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row)
                );
                CREATE TABLE IF NOT EXISTS attention_bias (
                    idx INTEGER,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row)
                );
                CREATE TABLE IF NOT EXISTS specialization_weights (
                    idx INTEGER,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row, col)
                );
                CREATE TABLE IF NOT EXISTS attention_memory (
                    idx INTEGER,
                    window INTEGER,
                    seq INTEGER,
                    dim INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, window, seq, dim)
                );
                CREATE TABLE IF NOT EXISTS transformers (
                    idx INTEGER,
                    layer INTEGER,
                    weight_type TEXT,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, weight_type, row, col)
                );
                CREATE TABLE IF NOT EXISTS transformer_biases (
                    idx INTEGER,
                    layer INTEGER,
                    bias_type TEXT,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, bias_type, row)
                );
                CREATE TABLE IF NOT EXISTS transformer_layer_norm (
                    idx INTEGER,
                    layer INTEGER,
                    norm_type TEXT,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, norm_type, row)
                );
                CREATE TABLE IF NOT EXISTS momentum_weights (
                    idx INTEGER,
                    layer INTEGER,
                    weight_type TEXT,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, weight_type, row, col)
                );
                CREATE TABLE IF NOT EXISTS momentum_biases (
                    idx INTEGER,
                    layer INTEGER,
                    bias_type TEXT,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, bias_type, row)
                );
                CREATE TABLE IF NOT EXISTS gradient_accumulation (
                    idx INTEGER,
                    layer INTEGER,
                    weight_type TEXT,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, weight_type, row, col)
                );
                CREATE TABLE IF NOT EXISTS gradient_biases (
                    idx INTEGER,
                    layer INTEGER,
                    bias_type TEXT,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, bias_type, row)
                );
            `);

            const insertMetadata = db.prepare('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)');
            insertMetadata.run('trainingStepCount', this.#trainingStepCount.toString());
            insertMetadata.run('swarmIntelligenceFactor', this.#swarmIntelligenceFactor.toString());

            const insertEnsembleWeights = db.prepare('INSERT OR REPLACE INTO ensemble_weights (idx, weight) VALUES (?, ?)');
            this.#ensembleWeights.forEach((weight, idx) => {
                if (isValidNumber(weight)) {
                    insertEnsembleWeights.run(idx, weight);
                }
            });

            const insertPerformanceScores = db.prepare('INSERT OR REPLACE INTO performance_scores (idx, score) VALUES (?, ?)');
            this.#performanceScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertPerformanceScores.run(idx, score);
                }
            });

            const insertAgreementScores = db.prepare('INSERT OR REPLACE INTO agreement_scores (idx, score) VALUES (?, ?)');
            this.#agreementScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertAgreementScores.run(idx, score);
                }
            });

            const insertSpecializationScores = db.prepare('INSERT OR REPLACE INTO specialization_scores (idx, score) VALUES (?, ?)');
            this.#specializationScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertSpecializationScores.run(idx, score);
                }
            });

            const insertHistoricalPerformance = db.prepare('INSERT OR REPLACE INTO historical_performance (idx, step, score) VALUES (?, ?, ?)');
            this.#historicalPerformance.forEach((history, idx) => {
                history.forEach((score, step) => {
                    if (isValidNumber(score)) {
                        insertHistoricalPerformance.run(idx, step, score);
                    }
                });
            });

            const insertTrustScoresHistory = db.prepare('INSERT OR REPLACE INTO trust_scores_history (idx, step, score) VALUES (?, ?, ?)');
            this.#trustScoresHistory.forEach((history, idx) => {
                history.forEach((score, step) => {
                    if (isValidNumber(score)) {
                        insertTrustScoresHistory.run(idx, step, score);
                    }
                });
            });

            const insertAdaptiveLearningRate = db.prepare('INSERT OR REPLACE INTO adaptive_learning_rate (idx, rate) VALUES (?, ?)');
            this.#adaptiveLearningRate.forEach((rate, idx) => {
                if (isValidNumber(rate)) {
                    insertAdaptiveLearningRate.run(idx, rate);
                }
            });

            const insertAttentionWeightMatrix = db.prepare('INSERT OR REPLACE INTO attention_weight_matrix (idx, row, value) VALUES (?, ?, ?)');
            this.#attentionWeightMatrix.forEach((weights, idx) => {
                weights.forEach((value, row) => {
                    if (isValidNumber(value)) {
                        insertAttentionWeightMatrix.run(idx, row, value);
                    }
                });
            });

            const insertAttentionBias = db.prepare('INSERT OR REPLACE INTO attention_bias (idx, row, value) VALUES (?, ?, ?)');
            this.#attentionBias.forEach((biases, idx) => {
                biases.forEach((value, row) => {
                    if (isValidNumber(value)) {
                        insertAttentionBias.run(idx, row, value);
                    }
                });
            });

            const insertSpecializationWeights = db.prepare('INSERT OR REPLACE INTO specialization_weights (idx, row, col, value) VALUES (?, ?, ?, ?)');
            this.#specializationWeights.forEach((matrix, idx) => {
                matrix.forEach((row, r) => {
                    row.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertSpecializationWeights.run(idx, r, c, value);
                        }
                    });
                });
            });

            const insertAttentionMemory = db.prepare('INSERT OR REPLACE INTO attention_memory (idx, window, seq, dim, value) VALUES (?, ?, ?, ?, ?)');
            this.#attentionMemory.forEach((memory, idx) => {
                memory.forEach((window, w) => {
                    window.forEach((seq, s) => {
                        seq.forEach((value, d) => {
                            if (isValidNumber(value)) {
                                insertAttentionMemory.run(idx, w, s, d, value);
                            }
                        });
                    });
                });
            });

            const insertTransformerWeights = db.prepare('INSERT OR REPLACE INTO transformers (idx, layer, weight_type, row, col, value) VALUES (?, ?, ?, ?, ?, ?)');
            const insertTransformerBiases = db.prepare('INSERT OR REPLACE INTO transformer_biases (idx, layer, bias_type, row, value) VALUES (?, ?, ?, ?, ?)');
            const insertTransformerLayerNorm = db.prepare('INSERT OR REPLACE INTO transformer_layer_norm (idx, layer, norm_type, row, value) VALUES (?, ?, ?, ?, ?)');
            this.#transformers.forEach((transformer, idx) => {
                transformer.attentionWeights.forEach((layer, l) => {
                    layer.Wq.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'Wq', r, c, value);
                            }
                        });
                    });
                    layer.Wk.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'Wk', r, c, value);
                            }
                        });
                    });
                    layer.Wv.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'Wv', r, c, value);
                            }
                        });
                    });
                    layer.Wo.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'Wo', r, c, value);
                            }
                        });
                    });
                });
                transformer.ffnWeights.forEach((layer, l) => {
                    layer.W1.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'W1', r, c, value);
                            }
                        });
                    });
                    layer.W2.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'W2', r, c, value);
                            }
                        });
                    });
                    layer.b1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerBiases.run(idx, l, 'b1', r, value);
                        }
                    });
                    layer.b2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerBiases.run(idx, l, 'b2', r, value);
                        }
                    });
                });
                transformer.layerNormWeights.forEach((layer, l) => {
                    layer.gamma1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerLayerNorm.run(idx, l, 'gamma1', r, value);
                        }
                    });
                    layer.beta1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerLayerNorm.run(idx, l, 'beta1', r, value);
                        }
                    });
                    layer.gamma2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerLayerNorm.run(idx, l, 'gamma2', r, value);
                        }
                    });
                    layer.beta2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerLayerNorm.run(idx, l, 'beta2', r, value);
                        }
                    });
                });
                transformer.outputWeights.forEach((row, r) => {
                    row.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertTransformerWeights.run(idx, -1, 'outputWeights', r, c, value);
                        }
                    });
                });
                transformer.outputBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertTransformerBiases.run(idx, -1, 'outputBias', r, value);
                    }
                });
            });

            const insertMomentumWeights = db.prepare('INSERT OR REPLACE INTO momentum_weights (idx, layer, weight_type, row, col, value) VALUES (?, ?, ?, ?, ?, ?)');
            const insertMomentumBiases = db.prepare('INSERT OR REPLACE INTO momentum_biases (idx, layer, bias_type, row, value) VALUES (?, ?, ?, ?, ?)');
            this.#momentumWeights.forEach((momentum, idx) => {
                momentum.attentionWeights.forEach((layer, l) => {
                    layer.Wq.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertMomentumWeights.run(idx, l, 'Wq', r, c, value);
                            }
                        });
                    });
                    layer.Wk.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertMomentumWeights.run(idx, l, 'Wk', r, c, value);
                            }
                        });
                    });
                    layer.Wv.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertMomentumWeights.run(idx, l, 'Wv', r, c, value);
                            }
                        });
                    });
                    layer.Wo.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertMomentumWeights.run(idx, l, 'Wo', r, c, value);
                            }
                        });
                    });
                });
                momentum.ffnWeights.forEach((layer, l) => {
                    layer.W1.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertMomentumWeights.run(idx, l, 'W1', r, c, value);
                            }
                        });
                    });
                    layer.W2.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertMomentumWeights.run(idx, l, 'W2', r, c, value);
                            }
                        });
                    });
                    layer.b1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertMomentumBiases.run(idx, l, 'b1', r, value);
                        }
                    });
                    layer.b2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertMomentumBiases.run(idx, l, 'b2', r, value);
                        }
                    });
                });
                momentum.layerNormWeights.forEach((layer, l) => {
                    layer.gamma1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertMomentumBiases.run(idx, l, 'gamma1', r, value);
                        }
                    });
                    layer.beta1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertMomentumBiases.run(idx, l, 'beta1', r, value);
                        }
                    });
                    layer.gamma2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertMomentumBiases.run(idx, l, 'gamma2', r, value);
                        }
                    });
                    layer.beta2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertMomentumBiases.run(idx, l, 'beta2', r, value);
                        }
                    });
                });
                momentum.outputWeights.forEach((row, r) => {
                    row.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertMomentumWeights.run(idx, -1, 'outputWeights', r, c, value);
                        }
                    });
                });
                momentum.outputBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertMomentumBiases.run(idx, -1, 'outputBias', r, value);
                    }
                });
            });

            const insertGradientAccumulation = db.prepare('INSERT OR REPLACE INTO gradient_accumulation (idx, layer, weight_type, row, col, value) VALUES (?, ?, ?, ?, ?, ?)');
            const insertGradientBiases = db.prepare('INSERT OR REPLACE INTO gradient_biases (idx, layer, bias_type, row, value) VALUES (?, ?, ?, ?, ?)');
            this.#gradientAccumulation.forEach((gradient, idx) => {
                gradient.attentionWeights.forEach((layer, l) => {
                    layer.Wq.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'Wq', r, c, value);
                            }
                        });
                    });
                    layer.Wk.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'Wk', r, c, value);
                            }
                        });
                    });
                    layer.Wv.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'Wv', r, c, value);
                            }
                        });
                    });
                    layer.Wo.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'Wo', r, c, value);
                            }
                        });
                    });
                });
                gradient.ffnWeights.forEach((layer, l) => {
                    layer.W1.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'W1', r, c, value);
                            }
                        });
                    });
                    layer.W2.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'W2', r, c, value);
                            }
                        });
                    });
                    layer.b1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientBiases.run(idx, l, 'b1', r, value);
                        }
                    });
                    layer.b2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientBiases.run(idx, l, 'b2', r, value);
                        }
                    });
                });
                gradient.layerNormWeights.forEach((layer, l) => {
                    layer.gamma1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientBiases.run(idx, l, 'gamma1', r, value);
                        }
                    });
                    layer.beta1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientBiases.run(idx, l, 'beta1', r, value);
                        }
                    });
                    layer.gamma2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientBiases.run(idx, l, 'gamma2', r, value);
                        }
                    });
                    layer.beta2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientBiases.run(idx, l, 'beta2', r, value);
                        }
                    });
                });
                gradient.outputWeights.forEach((row, r) => {
                    row.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, -1, 'outputWeights', r, c, value);
                        }
                    });
                });
                gradient.outputBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertGradientBiases.run(idx, -1, 'outputBias', r, value);
                    }
                });
            });

            db.exec('COMMIT');
        } catch (error) {
            db.exec('ROLLBACK');
        } finally {
            if (db) {
                db.close();
            }
        }
    }

    #loadState() {
        const dbPath = path.join(directoryPath, 'hivemind_state.db');
        let db;

        try {
            if (!fs.existsSync(dbPath)) {
                return;
            }

            db = new Database(dbPath, { readonly: true });

            const metadataStmt = db.prepare('SELECT key, value FROM metadata WHERE key = ?');
            const trainingStepCount = metadataStmt.get('trainingStepCount');
            if (trainingStepCount && isValidNumber(Number(trainingStepCount.value))) {
                this.#trainingStepCount = Number(trainingStepCount.value);
            }
            const swarmIntelligenceFactor = metadataStmt.get('swarmIntelligenceFactor');
            if (swarmIntelligenceFactor && isValidNumber(Number(swarmIntelligenceFactor.value))) {
                this.#swarmIntelligenceFactor = Number(swarmIntelligenceFactor.value);
            }

            const ensembleWeightsStmt = db.prepare('SELECT idx, weight FROM ensemble_weights');
            const ensembleWeights = ensembleWeightsStmt.all();
            ensembleWeights.forEach(({ idx, weight }) => {
                if (isValidNumber(weight) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#ensembleWeights[idx] = weight;
                }
            });
            this.#normalizeEnsembleWeights();

            const performanceScoresStmt = db.prepare('SELECT idx, score FROM performance_scores');
            const performanceScores = performanceScoresStmt.all();
            performanceScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#performanceScores[idx] = score;
                }
            });

            const agreementScoresStmt = db.prepare('SELECT idx, score FROM agreement_scores');
            const agreementScores = agreementScoresStmt.all();
            agreementScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#agreementScores[idx] = score;
                }
            });

            const specializationScoresStmt = db.prepare('SELECT idx, score FROM specialization_scores');
            const specializationScores = specializationScoresStmt.all();
            specializationScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#specializationScores[idx] = score;
                }
            });

            const historicalPerformanceStmt = db.prepare('SELECT idx, step, score FROM historical_performance ORDER BY idx, step');
            const historicalPerformance = historicalPerformanceStmt.all();
            this.#historicalPerformance = Array(this.#ensembleSize).fill().map(() => []);
            historicalPerformance.forEach(({ idx, step, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize && Number.isInteger(step)) {
                    this.#historicalPerformance[idx][step] = score;
                }
            });

            const trustScoresHistoryStmt = db.prepare('SELECT idx, step, score FROM trust_scores_history ORDER BY idx, step');
            const trustScoresHistory = trustScoresHistoryStmt.all();
            this.#trustScoresHistory = Array(this.#ensembleSize).fill().map(() => []);
            trustScoresHistory.forEach(({ idx, step, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize && Number.isInteger(step)) {
                    this.#trustScoresHistory[idx][step] = score;
                }
            });

            const adaptiveLearningRateStmt = db.prepare('SELECT idx, rate FROM adaptive_learning_rate');
            const adaptiveLearningRates = adaptiveLearningRateStmt.all();
            adaptiveLearningRates.forEach(({ idx, rate }) => {
                if (isValidNumber(rate) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#adaptiveLearningRate[idx] = rate;
                }
            });

            const attentionWeightMatrixStmt = db.prepare('SELECT idx, row, value FROM attention_weight_matrix');
            const attentionWeightMatrix = attentionWeightMatrixStmt.all();
            attentionWeightMatrix.forEach(({ idx, row, value }) => {
                if (isValidNumber(value) && idx >= 0 && idx < this.#ensembleSize && row >= 0 && row < this.#hiddenSize) {
                    this.#attentionWeightMatrix[idx][row] = value;
                }
            });

            const attentionBiasStmt = db.prepare('SELECT idx, row, value FROM attention_bias');
            const attentionBias = attentionBiasStmt.all();
            attentionBias.forEach(({ idx, row, value }) => {
                if (isValidNumber(value) && idx >= 0 && idx < this.#ensembleSize && row >= 0 && row < this.#hiddenSize) {
                    this.#attentionBias[idx][row] = value;
                }
            });

            const specializationWeightsStmt = db.prepare('SELECT idx, row, col, value FROM specialization_weights');
            const specializationWeights = specializationWeightsStmt.all();
            specializationWeights.forEach(({ idx, row, col, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0 && row < this.#hiddenSize &&
                    col >= 0 && col < this.#hiddenSize
                ) {
                    this.#specializationWeights[idx][row][col] = value;
                }
            });

            const attentionMemoryStmt = db.prepare('SELECT idx, window, seq, dim, value FROM attention_memory');
            const attentionMemory = attentionMemoryStmt.all();
            this.#attentionMemory = Array(this.#ensembleSize).fill().map(() =>
                Array(this.#contextWindow).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)))
            );
            attentionMemory.forEach(({ idx, window, seq, dim, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    window >= 0 && window < this.#contextWindow &&
                    seq >= 0 && seq < this.#inputSize &&
                    dim >= 0 && dim < this.#hiddenSize
                ) {
                    this.#attentionMemory[idx][window][seq][dim] = value;
                }
            });

            const transformerWeightsStmt = db.prepare('SELECT idx, layer, weight_type, row, col, value FROM transformers');
            const transformerBiasesStmt = db.prepare('SELECT idx, layer, bias_type, row, value FROM transformer_biases');
            const transformerLayerNormStmt = db.prepare('SELECT idx, layer, norm_type, row, value FROM transformer_layer_norm');
            const transformerWeights = transformerWeightsStmt.all();
            const transformerBiases = transformerBiasesStmt.all();
            const transformerLayerNorm = transformerLayerNormStmt.all();
            transformerWeights.forEach(({ idx, layer, weight_type, row, col, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0 && col >= 0
                ) {
                    if (weight_type === 'outputWeights' && layer === -1) {
                        if (row < this.#hiddenSize && col < this.#outputSize) {
                            this.#transformers[idx].outputWeights[row][col] = value;
                        }
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (weight_type === 'Wq' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer].Wq[row][col] = value;
                        } else if (weight_type === 'Wk' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer].Wk[row][col] = value;
                        } else if (weight_type === 'Wv' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer].Wv[row][col] = value;
                        } else if (weight_type === 'Wo' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer].Wo[row][col] = value;
                        } else if (weight_type === 'W1' && row < this.#hiddenSize && col < this.#feedForwardSize) {
                            this.#transformers[idx].ffnWeights[layer].W1[row][col] = value;
                        } else if (weight_type === 'W2' && row < this.#feedForwardSize && col < this.#hiddenSize) {
                            this.#transformers[idx].ffnWeights[layer].W2[row][col] = value;
                        }
                    }
                }
            });
            transformerBiases.forEach(({ idx, layer, bias_type, row, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0
                ) {
                    if (bias_type === 'outputBias' && layer === -1 && row < this.#outputSize) {
                        this.#transformers[idx].outputBias[row] = value;
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (bias_type === 'b1' && row < this.#feedForwardSize) {
                            this.#transformers[idx].ffnWeights[layer].b1[row] = value;
                        } else if (bias_type === 'b2' && row < this.#hiddenSize) {
                            this.#transformers[idx].ffnWeights[layer].b2[row] = value;
                        }
                    }
                }
            });
            transformerLayerNorm.forEach(({ idx, layer, norm_type, row, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    layer >= 0 && layer < this.#numLayers &&
                    row >= 0 && row < this.#hiddenSize
                ) {
                    if (norm_type === 'gamma1') {
                        this.#transformers[idx].layerNormWeights[layer].gamma1[row] = value;
                    } else if (norm_type === 'beta1') {
                        this.#transformers[idx].layerNormWeights[layer].beta1[row] = value;
                    } else if (norm_type === 'gamma2') {
                        this.#transformers[idx].layerNormWeights[layer].gamma2[row] = value;
                    } else if (norm_type === 'beta2') {
                        this.#transformers[idx].layerNormWeights[layer].beta2[row] = value;
                    }
                }
            });

            const momentumWeightsStmt = db.prepare('SELECT idx, layer, weight_type, row, col, value FROM momentum_weights');
            const momentumBiasesStmt = db.prepare('SELECT idx, layer, bias_type, row, value FROM momentum_biases');
            const momentumWeights = momentumWeightsStmt.all();
            const momentumBiases = momentumBiasesStmt.all();
            momentumWeights.forEach(({ idx, layer, weight_type, row, col, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0 && col >= 0
                ) {
                    if (weight_type === 'outputWeights' && layer === -1) {
                        if (row < this.#hiddenSize && col < this.#outputSize) {
                            this.#momentumWeights[idx].outputWeights[row][col] = value;
                        }
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (weight_type === 'Wq' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#momentumWeights[idx].attentionWeights[layer].Wq[row][col] = value;
                        } else if (weight_type === 'Wk' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#momentumWeights[idx].attentionWeights[layer].Wk[row][col] = value;
                        } else if (weight_type === 'Wv' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#momentumWeights[idx].attentionWeights[layer].Wv[row][col] = value;
                        } else if (weight_type === 'Wo' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#momentumWeights[idx].attentionWeights[layer].Wo[row][col] = value;
                        } else if (weight_type === 'W1' && row < this.#hiddenSize && col < this.#feedForwardSize) {
                            this.#momentumWeights[idx].ffnWeights[layer].W1[row][col] = value;
                        } else if (weight_type === 'W2' && row < this.#feedForwardSize && col < this.#hiddenSize) {
                            this.#momentumWeights[idx].ffnWeights[layer].W2[row][col] = value;
                        }
                    }
                }
            });
            momentumBiases.forEach(({ idx, layer, bias_type, row, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0
                ) {
                    if (bias_type === 'outputBias' && layer === -1 && row < this.#outputSize) {
                        this.#momentumWeights[idx].outputBias[row] = value;
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (bias_type === 'b1' && row < this.#feedForwardSize) {
                            this.#momentumWeights[idx].ffnWeights[layer].b1[row] = value;
                        } else if (bias_type === 'b2' && row < this.#hiddenSize) {
                            this.#momentumWeights[idx].ffnWeights[layer].b2[row] = value;
                        } else if (bias_type === 'gamma1' && row < this.#hiddenSize) {
                            this.#momentumWeights[idx].layerNormWeights[layer].gamma1[row] = value;
                        } else if (bias_type === 'beta1' && row < this.#hiddenSize) {
                            this.#momentumWeights[idx].layerNormWeights[layer].beta1[row] = value;
                        } else if (bias_type === 'gamma2' && row < this.#hiddenSize) {
                            this.#momentumWeights[idx].layerNormWeights[layer].gamma2[row] = value;
                        } else if (bias_type === 'beta2' && row < this.#hiddenSize) {
                            this.#momentumWeights[idx].layerNormWeights[layer].beta2[row] = value;
                        }
                    }
                }
            });

            const gradientAccumulationStmt = db.prepare('SELECT idx, layer, weight_type, row, col, value FROM gradient_accumulation');
            const gradientBiasesStmt = db.prepare('SELECT idx, layer, bias_type, row, value FROM gradient_biases');
            const gradientAccumulation = gradientAccumulationStmt.all();
            const gradientBiases = gradientBiasesStmt.all();
            gradientAccumulation.forEach(({ idx, layer, weight_type, row, col, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0 && col >= 0
                ) {
                    if (weight_type === 'outputWeights' && layer === -1) {
                        if (row < this.#hiddenSize && col < this.#outputSize) {
                            this.#gradientAccumulation[idx].outputWeights[row][col] = value;
                        }
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (weight_type === 'Wq' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wq[row][col] = value;
                        } else if (weight_type === 'Wk' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wk[row][col] = value;
                        } else if (weight_type === 'Wv' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wv[row][col] = value;
                        } else if (weight_type === 'Wo' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wo[row][col] = value;
                        } else if (weight_type === 'W1' && row < this.#hiddenSize && col < this.#feedForwardSize) {
                            this.#gradientAccumulation[idx].ffnWeights[layer].W1[row][col] = value;
                        } else if (weight_type === 'W2' && row < this.#feedForwardSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].ffnWeights[layer].W2[row][col] = value;
                        }
                    }
                }
            });
            gradientBiases.forEach(({ idx, layer, bias_type, row, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0
                ) {
                    if (bias_type === 'outputBias' && layer === -1 && row < this.#outputSize) {
                        this.#gradientAccumulation[idx].outputBias[row] = value;
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (bias_type === 'b1' && row < this.#feedForwardSize) {
                            this.#gradientAccumulation[idx].ffnWeights[layer].b1[row] = value;
                        } else if (bias_type === 'b2' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].ffnWeights[layer].b2[row] = value;
                        } else if (bias_type === 'gamma1' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[row] = value;
                        } else if (bias_type === 'beta1' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[row] = value;
                        } else if (bias_type === 'gamma2' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[row] = value;
                        } else if (bias_type === 'beta2' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[row] = value;
                        }
                    }
                }
            });

        } catch (error) {
        } finally {
            if (db) {
                db.close();
            }
        }
    }

    #xavierInit(rows, cols) {
        return Array(rows).fill().map(() =>
            Array(cols).fill().map(() => (Math.random() - 0.5) * Math.sqrt(2 / (rows + cols)))
        );
    }

    #createPositionalEncoding() {
        return Array(this.#inputSize).fill().map((_, pos) =>
            Array(this.#hiddenSize).fill().map((_, d) => {
                const exponent = 2 * Math.floor(d / 2) / this.#hiddenSize;
                const freq = 1 / (10000 ** exponent);
                return d % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
            })
        );
    }

    #createWeightStructure() {
        return Array(this.#ensembleSize).fill().map(() => ({
            outputWeights: Array(this.#hiddenSize).fill().map(() => Array(this.#outputSize).fill(0)),
            outputBias: Array(this.#outputSize).fill(0),
            attentionWeights: Array(this.#numLayers).fill().map(() => ({
                Wq: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wk: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wv: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wo: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
            })),
            ffnWeights: Array(this.#numLayers).fill().map(() => ({
                W1: Array(this.#hiddenSize).fill().map(() => Array(this.#feedForwardSize).fill(0)),
                W2: Array(this.#feedForwardSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                b1: Array(this.#feedForwardSize).fill(0),
                b2: Array(this.#hiddenSize).fill(0)
            })),
            layerNormWeights: Array(this.#numLayers).fill().map(() => ({
                gamma1: Array(this.#hiddenSize).fill(0),
                beta1: Array(this.#hiddenSize).fill(0),
                gamma2: Array(this.#hiddenSize).fill(0),
                beta2: Array(this.#hiddenSize).fill(0)
            })),
            attentionBias: Array(this.#hiddenSize).fill(0)
        }));
    }

    #shouldCommunicate() {
        if (
            !Array.isArray(this.#performanceScores) ||
            this.#performanceScores.length !== this.#ensembleSize ||
            this.#ensembleSize === 0 ||
            !this.#performanceScores.every(score => isValidNumber(score) && score >= 0 && score <= 1) ||
            !Array.isArray(this.#specializationScores) ||
            this.#specializationScores.length !== this.#ensembleSize ||
            !this.#specializationScores.every(score => isValidNumber(score) && score >= 0 && score <= 1)
        ) {
            return false;
        }

        const meanPerformance = this.#performanceScores.reduce((sum, score) => sum + score, 0) / this.#ensembleSize;

        if (this.#performanceScores.every(score => Math.abs(score - meanPerformance) < 1e-6)) {
            return false;
        }

        const variance = this.#performanceScores.reduce((sum, score) => sum + Math.pow(score - meanPerformance, 2), 0) / this.#ensembleSize;

        if (!isValidNumber(variance) || variance < 0) {
            return false;
        }

        const performanceStd = Math.sqrt(variance);

        const specializationMean = this.#specializationScores.reduce((sum, score) => sum + score, 0) / this.#ensembleSize;
        const specializationVariance = this.#specializationScores.reduce((sum, score) => sum + Math.pow(score - specializationMean, 2), 0) / this.#ensembleSize;
        const specializationStd = isValidNumber(specializationVariance) && specializationVariance >= 0 ? Math.sqrt(specializationVariance) : 0;

        const adjustedStd = performanceStd * (1 + this.#swarmIntelligenceFactor * specializationStd);

        const progress = Math.min(Math.max(this.#trainingStepCount / 5000, 0), 1);
        const threshold = 0.1 * (1 - progress) + 0.05;

        return adjustedStd > threshold;
    }

    #computeAttentionWeights(inputs, outputs) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber) ||
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber)
        ) {
            return Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
        }

        const inputFeatures = inputs.map(x =>
            Array(this.#hiddenSize).fill(isValidNumber(x) ? x : 0)
        );

        const attentionScores = Array(this.#ensembleSize).fill(0);
        for (let t = 0; t < this.#ensembleSize; t++) {
            if (
                !Array.isArray(this.#attentionMemory[t]) ||
                this.#attentionMemory[t].length === 0 ||
                !Array.isArray(this.#attentionMemory[t][0]) ||
                this.#attentionMemory[t][0].length !== this.#inputSize ||
                !this.#attentionMemory[t][0].every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
            ) {
                this.#attentionMemory[t] = Array(this.#contextWindow).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
                );
            }

            const queries = inputFeatures.map(row =>
                row.map((val, i) =>
                    Array(this.#hiddenSize).fill().reduce((sum, _, j) =>
                        sum + (
                            isValidNumber(val) && isValidNumber(this.#attentionWeightMatrix[t][j])
                                ? val * this.#attentionWeightMatrix[t][j]
                                : 0
                        ),
                        0
                    )
                )
            );

            const keys = inputFeatures.map(row =>
                row.map((val, i) =>
                    Array(this.#hiddenSize).fill().reduce((sum, _, j) =>
                        sum + (
                            isValidNumber(val) && isValidNumber(this.#attentionWeightMatrix[t][j])
                                ? val * this.#attentionWeightMatrix[t][j]
                                : 0
                        ),
                        0
                    )
                )
            );

            const values = Array(this.#inputSize).fill().map(() =>
                Array(this.#hiddenSize).fill(isValidNumber(outputs[t]) ? outputs[t] : 0)
            );

            let score = 0;
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#inputSize; j++) {
                    let dotProduct = 0;
                    for (let k = 0; k < this.#hiddenSize; k++) {
                        dotProduct += isValidNumber(queries[i][k]) && isValidNumber(keys[j][k])
                            ? queries[i][k] * keys[j][k]
                            : 0;
                    }
                    const attentionWeight = this.#softmax(
                        Array(this.#inputSize).fill().map((_, k) => {
                            let dp = 0;
                            for (let m = 0; m < this.#hiddenSize; m++) {
                                dp += isValidNumber(queries[i][m]) && isValidNumber(keys[k][m])
                                    ? queries[i][m] * keys[k][m]
                                    : 0;
                            }
                            return isValidNumber(dp)
                                ? dp / Math.sqrt(this.#hiddenSize) * this.#attentionScalingFactor
                                : 0;
                        })
                    )[j];

                    for (let k = 0; k < this.#hiddenSize; k++) {
                        score += isValidNumber(attentionWeight) && isValidNumber(values[j][k])
                            ? attentionWeight * values[j][k]
                            : 0;
                    }
                }
            }

            const biasSum = this.#attentionBias[t].reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0);
            score = score / this.#inputSize + (isValidNumber(biasSum) ? biasSum / this.#hiddenSize : 0);

            const performanceScore = isValidNumber(this.#performanceScores[t]) ? this.#performanceScores[t] : 0.5;
            const specializationBoost = 1 + (isValidNumber(this.#specializationScores[t])
                ? this.#specializationScores[t] * this.#swarmIntelligenceFactor
                : 0);

            attentionScores[t] = score * (0.7 + 0.2 * performanceScore + 0.1 * specializationBoost);
        }

        const weights = this.#softmax(attentionScores.map(score => isValidNumber(score) ? score : 0));

        const finalWeights = weights.map((w, idx) => {
            const performanceScore = isValidNumber(this.#performanceScores[idx]) ? this.#performanceScores[idx] : 0.5;
            const specializationFactor = 1 + (isValidNumber(this.#specializationScores[idx])
                ? this.#specializationScores[idx] * this.#swarmIntelligenceFactor
                : 0);
            const weight = 0.6 * w + 0.3 * performanceScore + 0.1 * specializationFactor;
            return isValidNumber(weight) && weight >= 0 ? weight : 1 / this.#ensembleSize;
        });

        const sum = finalWeights.reduce((s, w) => s + (isValidNumber(w) && w >= 0 ? w : 0), 0);
        if (sum <= 1e-6) {
            return Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
        }
        return finalWeights.map(w => (isValidNumber(w) && w >= 0 ? w / sum : 1 / this.#ensembleSize));
    }

    #updateTrustScores() {
        const performanceMean = this.#performanceScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0), 0) / this.#ensembleSize || 1;
        const performanceStd = Math.sqrt(
        this.#performanceScores.reduce((sum, score) => sum + ((isValidNumber(score) ? score : 0) - performanceMean) ** 2, 0) / this.#ensembleSize
        ) || 1;

        this.#trustScoresHistory = this.#trustScoresHistory.map((history, idx) => {
        const normalizedScore = isValidNumber(this.#performanceScores[idx]) && performanceStd > 0
            ? (this.#performanceScores[idx] - performanceMean) / performanceStd
            : 0;
        const agreementFactor = isValidNumber(this.#agreementScores[idx]) ? this.#agreementScores[idx] : 0.5;
        const historicalTrend = this.#historicalPerformance[idx].length > 1
            ? (this.#historicalPerformance[idx][this.#historicalPerformance[idx].length - 1] - this.#historicalPerformance[idx][0]) /
            this.#historicalPerformance[idx].length
            : 0;
        const specializationBoost = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
        const trustScore = this.#sigmoid(normalizedScore * (0.6 + 0.2 * agreementFactor + 0.1 * historicalTrend + 0.1 * specializationBoost));
        history.push(isValidNumber(trustScore) ? trustScore : 0.5);
        if (history.length > this.#maxTrustHistory) history.shift();
        return history;
        });

        const trustMomentum = 0.6;
        this.#performanceScores = this.#performanceScores.map((score, idx) => {
        const recentTrust = this.#trustScoresHistory[idx].slice(-5);
        const avgTrust = recentTrust.length > 0
            ? recentTrust.reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) / recentTrust.length
            : 0.5;
        const historicalWeight = this.#trustScoresHistory[idx].length / this.#maxTrustHistory;
        const specializationFactor = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
        return trustMomentum * (isValidNumber(score) ? score : 0) + (1 - trustMomentum) * avgTrust * (0.7 + 0.2 * historicalWeight + 0.1 * specializationFactor);
        });

        this.#ensembleWeights = this.#trustScoresHistory.map((history, idx) => {
        const recentTrust = history.slice(-5);
        const avgTrust = recentTrust.length > 0
            ? recentTrust.reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) / recentTrust.length
            : 1 / this.#ensembleSize;
        const specializationFactor = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
        return avgTrust * (0.8 + 0.2 * specializationFactor);
        });
        this.#normalizeEnsembleWeights();
    }

    #computeSpecializationScores(inputs, outputs) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber) ||
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber)
        ) {
            return;
        }

        const featureCorrelations = Array(this.#ensembleSize).fill().map(() => Array(this.#inputSize).fill(0));
        const inputMean = inputs.reduce((sum, v) => sum + (isValidNumber(v) ? v : 0), 0) / inputs.length || 0;
        const outputMean = outputs.reduce((sum, v) => sum + (isValidNumber(v) ? v : 0), 0) / outputs.length || 0;
        const inputStd = Math.sqrt(
            inputs.reduce((sum, v) => sum + ((isValidNumber(v) ? v : 0) - inputMean) ** 2, 0) / inputs.length
        ) || 1e-6;
        const outputStd = Math.sqrt(
            outputs.reduce((sum, v) => sum + ((isValidNumber(v) ? v : 0) - outputMean) ** 2, 0) / outputs.length
        ) || 1e-6;

        for (let i = 0; i < this.#ensembleSize; i++) {
            for (let j = 0; j < this.#inputSize; j++) {
                if (isValidNumber(inputs[j]) && isValidNumber(outputs[i])) {
                    const inputDiff = inputs[j] - inputMean;
                    const outputDiff = outputs[i] - outputMean;
                    const numerator = inputDiff * outputDiff;
                    featureCorrelations[i][j] = isValidNumber(numerator) && inputStd > 1e-6 && outputStd > 1e-6
                        ? Math.min(Math.max(numerator / (inputStd * outputStd), -1), 1)
                        : 0;
                }
            }
        }

        this.#specializationScores = featureCorrelations.map((corr, idx) => {
            const meanCorr = corr.reduce((sum, val) => sum + (isValidNumber(val) ? Math.abs(val) : 0), 0) / corr.length || 0.5;
            const performanceFactor = isValidNumber(this.#performanceScores[idx]) ? this.#performanceScores[idx] : 0.5;
            const outputVariance = Math.abs(outputs[idx] - outputMean) / (outputStd || 1e-6);
            const diversityBoost = 0.3 * outputVariance;
            return this.#sigmoid(meanCorr * (1 + this.#swarmIntelligenceFactor) * (0.5 + 0.3 * performanceFactor + diversityBoost));
        });

        const maxScore = Math.max(...this.#specializationScores.map(score => isValidNumber(score) ? score : 0)) || 1;
        this.#specializationScores = this.#specializationScores.map(score => isValidNumber(score) ? score / maxScore : 0);

        for (let i = 0; i < this.#ensembleSize; i++) {
            const specializationFactor = Math.min(Math.max(1 + this.#specializationScores[i] * this.#swarmIntelligenceFactor * 0.5, 0.5), 1.5);
            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const inputIdx = (j + k) % this.#inputSize;
                    const corr = featureCorrelations[i][inputIdx];
                    if (isValidNumber(corr)) {
                        const update = isValidNumber(this.#adaptiveLearningRate[i]) && isValidNumber(corr)
                            ? Math.min(Math.max(this.#adaptiveLearningRate[i] * corr * specializationFactor * 0.1, -0.005), 0.005)
                            : 0;
                        this.#specializationWeights[i][j][k] += update;
                        this.#specializationWeights[i][j][k] = Math.min(Math.max(this.#specializationWeights[i][j][k], -0.5), 0.5);
                    }
                }
            }
        }
    }

    #updateAdaptiveLearningRates() {
        if (
            !Array.isArray(this.#performanceScores) ||
            this.#performanceScores.length !== this.#ensembleSize ||
            !this.#performanceScores.every(score => isValidNumber(score) && score >= 0 && score <= 1)
        ) {
            this.#adaptiveLearningRate = Array(this.#ensembleSize).fill(this.#learningRate);
            return;
        }

        const performanceMean = this.#performanceScores.reduce((sum, score) => sum + score, 0) / this.#ensembleSize;
        const performanceStd = Math.sqrt(
            this.#performanceScores.reduce((sum, score) => sum + ((isValidNumber(score) ? score : 0) - performanceMean) ** 2, 0) / this.#ensembleSize
        ) || 0.1;

        this.#adaptiveLearningRate = this.#adaptiveLearningRate.map((lr, idx) => {
            const performanceDiff = isValidNumber(this.#performanceScores[idx])
                ? (this.#performanceScores[idx] - performanceMean) / (performanceStd + 1e-6)
                : 0;
            const adjustment = 1 + 0.5 * this.#sigmoid(performanceDiff * 2);
            const newLr = this.#learningRate * adjustment;
            return Math.min(Math.max(newLr, this.#learningRate * 0.1), this.#learningRate * 5);
        });
    }

    #shareWeights() {
        const performanceSum = this.#performanceScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0), 0) || 1;
        const trustScores = this.#performanceScores.map(score => (isValidNumber(score) ? score : 0) / performanceSum);
        const momentumFactor = 0.7;

        const avgWeights = (weights, trustWeights) => {
            const result = weights[0].map(row => row.map(() => 0));
            for (let i = 0; i < result.length; i++) {
                for (let j = 0; j < result[i].length; j++) {
                    let weightedSum = 0, totalWeight = 0;
                    for (let t = 0; t < weights.length; t++) {
                        if (isValidNumber(weights[t][i][j]) && isValidNumber(trustWeights[t])) {
                            weightedSum += weights[t][i][j] * trustWeights[t] * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[t]);
                            totalWeight += trustWeights[t];
                        }
                    }
                    result[i][j] = totalWeight > 1e-6 ? weightedSum / totalWeight : 0;
                }
            }
            return result;
        };

        const avgBias = (biases, trustWeights) => {
            const result = biases[0].map(() => 0);
            for (let i = 0; i < result.length; i++) {
                let weightedSum = 0, totalWeight = 0;
                for (let t = 0; t < biases.length; t++) {
                    if (isValidNumber(biases[t][i]) && isValidNumber(trustWeights[t])) {
                        weightedSum += biases[t][i] * trustWeights[t] * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[t]);
                        totalWeight += trustWeights[t];
                    }
                }
                result[i] = totalWeight > 1e-6 ? weightedSum / totalWeight : 0;
            }
            return result;
        };

        const topPerformers = this.#performanceScores
            .map((score, idx) => ({ score: isValidNumber(score) ? score : 0, idx }))
            .sort((a, b) => b.score - a.score)
            .slice(0, Math.floor(this.#ensembleSize * 0.25))
            .map(({ idx }) => idx);

        for (let layer = 0; layer < this.#numLayers; layer++) {
            const allWq = this.#transformers.map(t => t.attentionWeights[layer].Wq);
            const allWk = this.#transformers.map(t => t.attentionWeights[layer].Wk);
            const allWv = this.#transformers.map(t => t.attentionWeights[layer].Wv);
            const allWo = this.#transformers.map(t => t.attentionWeights[layer].Wo);
            const allW1 = this.#transformers.map(t => t.ffnWeights[layer].W1);
            const allW2 = this.#transformers.map(t => t.ffnWeights[layer].W2);
            const allB1 = this.#transformers.map(t => t.ffnWeights[layer].b1);
            const allB2 = this.#transformers.map(t => t.ffnWeights[layer].b2);
            const allGamma1 = this.#transformers.map(t => t.layerNormWeights[layer].gamma1);
            const allBeta1 = this.#transformers.map(t => t.layerNormWeights[layer].beta1);
            const allGamma2 = this.#transformers.map(t => t.layerNormWeights[layer].gamma2);
            const allBeta2 = this.#transformers.map(t => t.layerNormWeights[layer].beta2);

            const avgWq = avgWeights(allWq, trustScores);
            const avgWk = avgWeights(allWk, trustScores);
            const avgWv = avgWeights(allWv, trustScores);
            const avgWo = avgWeights(allWo, trustScores);
            const avgW1 = avgWeights(allW1, trustScores);
            const avgW2 = avgWeights(allW2, trustScores);
            const avgB1 = avgBias(allB1, trustScores);
            const avgB2 = avgBias(allB2, trustScores);
            const avgGamma1 = avgBias(allGamma1, trustScores);
            const avgBeta1 = avgBias(allBeta1, trustScores);
            const avgGamma2 = avgBias(allGamma2, trustScores);
            const avgBeta2 = avgBias(allBeta2, trustScores);

            this.#transformers.forEach((t, idx) => {
                if (topPerformers.includes(idx)) return;

                const sharingRate = trustScores[idx] < 0.5 ? this.#weightSharingRate + 0.1 * (0.5 - trustScores[idx]) : this.#weightSharingRate;
                const swarmInfluence = 1 + this.#swarmIntelligenceFactor * (isValidNumber(this.#specializationScores[idx]) ? this.#specializationScores[idx] : 0);

                for (let i = 0; i < t.attentionWeights[layer].Wq.length; i++) {
                    for (let j = 0; j < t.attentionWeights[layer].Wq[i].length; j++) {
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                            ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j]
                            : 1;
                        const update = isValidNumber(t.attentionWeights[layer].Wq[i][j]) && isValidNumber(avgWq[i][j])
                            ? (1 - sharingRate) * t.attentionWeights[layer].Wq[i][j] + sharingRate * avgWq[i][j] * swarmInfluence * specializationFactor
                            : t.attentionWeights[layer].Wq[i][j] || 0;
                        this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j] +
                            (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                        t.attentionWeights[layer].Wq[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j];

                        const wkUpdate = isValidNumber(t.attentionWeights[layer].Wk[i][j]) && isValidNumber(avgWk[i][j])
                            ? (1 - sharingRate) * t.attentionWeights[layer].Wk[i][j] + sharingRate * avgWk[i][j] * swarmInfluence * specializationFactor
                            : t.attentionWeights[layer].Wk[i][j] || 0;
                        this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j] +
                            (1 - momentumFactor) * Math.min(Math.max(wkUpdate, -1), 1);
                        t.attentionWeights[layer].Wk[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j];

                        const wvUpdate = isValidNumber(t.attentionWeights[layer].Wv[i][j]) && isValidNumber(avgWv[i][j])
                            ? (1 - sharingRate) * t.attentionWeights[layer].Wv[i][j] + sharingRate * avgWv[i][j] * swarmInfluence * specializationFactor
                            : t.attentionWeights[layer].Wv[i][j] || 0;
                        this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j] +
                            (1 - momentumFactor) * Math.min(Math.max(wvUpdate, -1), 1);
                        t.attentionWeights[layer].Wv[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j];

                        const woUpdate = isValidNumber(t.attentionWeights[layer].Wo[i][j]) && isValidNumber(avgWo[i][j])
                            ? (1 - sharingRate) * t.attentionWeights[layer].Wo[i][j] + sharingRate * avgWo[i][j] * swarmInfluence * specializationFactor
                            : t.attentionWeights[layer].Wo[i][j] || 0;
                        this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j] +
                            (1 - momentumFactor) * Math.min(Math.max(woUpdate, -1), 1);
                        t.attentionWeights[layer].Wo[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j];
                    }
                }

                for (let i = 0; i < t.ffnWeights[layer].W1.length; i++) {
                    for (let j = 0; j < t.ffnWeights[layer].W1[i].length; j++) {
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                            ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize]
                            : 1;
                        const update = isValidNumber(t.ffnWeights[layer].W1[i][j]) && isValidNumber(avgW1[i][j])
                            ? (1 - sharingRate) * t.ffnWeights[layer].W1[i][j] + sharingRate * avgW1[i][j] * swarmInfluence * specializationFactor
                            : t.ffnWeights[layer].W1[i][j] || 0;
                        this.#momentumWeights[idx].ffnWeights[layer].W1[i][j] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].W1[i][j] +
                            (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                        t.ffnWeights[layer].W1[i][j] = this.#momentumWeights[idx].ffnWeights[layer].W1[i][j];
                    }
                }
                for (let i = 0; i < t.ffnWeights[layer].W2.length; i++) {
                    for (let j = 0; j < t.ffnWeights[layer].W2[i].length; j++) {
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize])
                            ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize]
                            : 1;
                        const update = isValidNumber(t.ffnWeights[layer].W2[i][j]) && isValidNumber(avgW2[i][j])
                            ? (1 - sharingRate) * t.ffnWeights[layer].W2[i][j] + sharingRate * avgW2[i][j] * swarmInfluence * specializationFactor
                            : t.ffnWeights[layer].W2[i][j] || 0;
                        this.#momentumWeights[idx].ffnWeights[layer].W2[i][j] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].W2[i][j] +
                            (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                        t.ffnWeights[layer].W2[i][j] = this.#momentumWeights[idx].ffnWeights[layer].W2[i][j];
                    }
                }

                for (let i = 0; i < t.ffnWeights[layer].b1.length; i++) {
                    const update = isValidNumber(t.ffnWeights[layer].b1[i]) && isValidNumber(avgB1[i])
                        ? (1 - sharingRate) * t.ffnWeights[layer].b1[i] + sharingRate * avgB1[i] * swarmInfluence
                        : t.ffnWeights[layer].b1[i] || 0;
                    this.#momentumWeights[idx].ffnWeights[layer].b1[i] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].b1[i] +
                        (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                    t.ffnWeights[layer].b1[i] = this.#momentumWeights[idx].ffnWeights[layer].b1[i];
                }
                for (let i = 0; i < t.ffnWeights[layer].b2.length; i++) {
                    const update = isValidNumber(t.ffnWeights[layer].b2[i]) && isValidNumber(avgB2[i])
                        ? (1 - sharingRate) * t.ffnWeights[layer].b2[i] + sharingRate * avgB2[i] * swarmInfluence
                        : t.ffnWeights[layer].b2[i] || 0;
                    this.#momentumWeights[idx].ffnWeights[layer].b2[i] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].b2[i] +
                        (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                    t.ffnWeights[layer].b2[i] = this.#momentumWeights[idx].ffnWeights[layer].b2[i];
                }

                for (let i = 0; i < t.layerNormWeights[layer].gamma1.length; i++) {
                    const gamma1Update = isValidNumber(t.layerNormWeights[layer].gamma1[i]) && isValidNumber(avgGamma1[i])
                        ? (1 - sharingRate) * t.layerNormWeights[layer].gamma1[i] + sharingRate * avgGamma1[i] * swarmInfluence
                        : t.layerNormWeights[layer].gamma1[i] || 1;
                    this.#momentumWeights[idx].layerNormWeights[layer].gamma1[i] = momentumFactor * this.#momentumWeights[idx].layerNormWeights[layer].gamma1[i] +
                        (1 - momentumFactor) * Math.min(Math.max(gamma1Update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
                    t.layerNormWeights[layer].gamma1[i] = this.#momentumWeights[idx].layerNormWeights[layer].gamma1[i];

                    const beta1Update = isValidNumber(t.layerNormWeights[layer].beta1[i]) && isValidNumber(avgBeta1[i])
                        ? (1 - sharingRate) * t.layerNormWeights[layer].beta1[i] + sharingRate * avgBeta1[i] * swarmInfluence
                        : t.layerNormWeights[layer].beta1[i] || 0;
                    this.#momentumWeights[idx].layerNormWeights[layer].beta1[i] = momentumFactor * this.#momentumWeights[idx].layerNormWeights[layer].beta1[i] +
                        (1 - momentumFactor) * Math.min(Math.max(beta1Update, -1), 1);
                    t.layerNormWeights[layer].beta1[i] = this.#momentumWeights[idx].layerNormWeights[layer].beta1[i];

                    const gamma2Update = isValidNumber(t.layerNormWeights[layer].gamma2[i]) && isValidNumber(avgGamma2[i])
                        ? (1 - sharingRate) * t.layerNormWeights[layer].gamma2[i] + sharingRate * avgGamma2[i] * swarmInfluence
                        : t.layerNormWeights[layer].gamma2[i] || 1;
                    this.#momentumWeights[idx].layerNormWeights[layer].gamma2[i] = momentumFactor * this.#momentumWeights[idx].layerNormWeights[layer].gamma2[i] +
                        (1 - momentumFactor) * Math.min(Math.max(gamma2Update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
                    t.layerNormWeights[layer].gamma2[i] = this.#momentumWeights[idx].layerNormWeights[layer].gamma2[i];

                    const beta2Update = isValidNumber(t.layerNormWeights[layer].beta2[i]) && isValidNumber(avgBeta2[i])
                        ? (1 - sharingRate) * t.layerNormWeights[layer].beta2[i] + sharingRate * avgBeta2[i] * swarmInfluence
                        : t.layerNormWeights[layer].beta2[i] || 0;
                    this.#momentumWeights[idx].layerNormWeights[layer].beta2[i] = momentumFactor * this.#momentumWeights[idx].layerNormWeights[layer].beta2[i] +
                        (1 - momentumFactor) * Math.min(Math.max(beta2Update, -1), 1);
                    t.layerNormWeights[layer].beta2[i] = this.#momentumWeights[idx].layerNormWeights[layer].beta2[i];
                }
            });
        }

        const allOutputWeights = this.#transformers.map(t => t.outputWeights);
        const allOutputBias = this.#transformers.map(t => t.outputBias);
        const avgOutputWeights = avgWeights(allOutputWeights, trustScores);
        const avgOutputBias = avgBias(allOutputBias, trustScores);

        this.#transformers.forEach((t, idx) => {
            if (topPerformers.includes(idx)) return;

            const sharingRate = trustScores[idx] < 0.5 ? this.#weightSharingRate + 0.1 * (0.5 - trustScores[idx]) : this.#weightSharingRate;
            const swarmInfluence = 1 + this.#swarmIntelligenceFactor * (isValidNumber(this.#specializationScores[idx]) ? this.#specializationScores[idx] : 0);

            for (let i = 0; i < t.outputWeights.length; i++) {
                for (let j = 0; j < t.outputWeights[i].length; j++) {
                    const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                        ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j]
                        : 1;
                    const update = isValidNumber(t.outputWeights[i][j]) && isValidNumber(avgOutputWeights[i][j])
                        ? (1 - sharingRate) * t.outputWeights[i][j] + sharingRate * avgOutputWeights[i][j] * swarmInfluence * specializationFactor
                        : t.outputWeights[i][j] || 0;
                    this.#momentumWeights[idx].outputWeights[i][j] = momentumFactor * this.#momentumWeights[idx].outputWeights[i][j] +
                        (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                    t.outputWeights[i][j] = this.#momentumWeights[idx].outputWeights[i][j];
                }
            }

            for (let i = 0; i < t.outputBias.length; i++) {
                const update = isValidNumber(t.outputBias[i]) && isValidNumber(avgOutputBias[i])
                    ? (1 - sharingRate) * t.outputBias[i] + sharingRate * avgOutputBias[i] * swarmInfluence
                    : t.outputBias[i] || 0;
                this.#momentumWeights[idx].outputBias[i] = momentumFactor * this.#momentumWeights[idx].outputBias[i] +
                    (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                t.outputBias[i] = this.#momentumWeights[idx].outputBias[i];
            }
        });
    }

    #normalizeEnsembleWeights() {
        const sum = this.#ensembleWeights.reduce((s, w) => s + (isValidNumber(w) && w >= 0 ? w : 0), 0);

        if (sum <= 1e-6) {
            this.#ensembleWeights = Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
            return;
        }

        this.#ensembleWeights = this.#ensembleWeights.map(w => 
            isValidNumber(w) && w >= 0 ? w / sum : 0
        );

        const finalSum = this.#ensembleWeights.reduce((s, w) => s + w, 0);
        if (Math.abs(finalSum - 1) > 1e-6) {
            const maxIndex = this.#ensembleWeights.indexOf(Math.max(...this.#ensembleWeights));
            this.#ensembleWeights[maxIndex] += 1 - finalSum;
        }
    }

    #normalizeAttentionWeights() {
        if (this.#trainingStepCount % 5000 !== 0) return;

        for (let i = 0; i < this.#ensembleSize; i++) {
            let norm = 0;
            for (let j = 0; j < this.#hiddenSize; j++) {
                if (isValidNumber(this.#attentionWeightMatrix[i][j])) {
                    norm += Math.pow(this.#attentionWeightMatrix[i][j], 2);
                }
            }
            norm = Math.sqrt(norm) || 1;
            for (let j = 0; j < this.#hiddenSize; j++) {
                if (isValidNumber(this.#attentionWeightMatrix[i][j])) {
                    this.#attentionWeightMatrix[i][j] = this.#attentionWeightMatrix[i][j] / norm;
                    this.#attentionWeightMatrix[i][j] = Math.min(Math.max(this.#attentionWeightMatrix[i][j], -1.0), 1.0);
                } else {
                    this.#attentionWeightMatrix[i][j] = 0;
                }
            }
        }
    }

    #adjustSwarmFactor(outputs) {
        if (this.#trainingStepCount % 1000 !== 0) return;

        if (
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber)
        ) {
            return;
        }

        let diversityLoss = this.#computeDiversityLoss(outputs);
        let normalizedDiversity = diversityLoss > 0.7 ? -1 : diversityLoss < 0.3 ? 1 : (0.5 - diversityLoss) / 0.2;

        let perfMean = this.#performanceScores.reduce((sum, x) => sum + (isValidNumber(x) ? x : 0), 0) / this.#ensembleSize;
        let perfVariance = Math.sqrt(
            this.#performanceScores.reduce((sum, x) => sum + ((isValidNumber(x) ? x : 0) - perfMean) ** 2, 0) / this.#ensembleSize
        ) || 0.1;
        let normalizedPerfVariance = perfVariance > 0.5 ? -1 : perfVariance < 0.2 ? 1 : (0.35 - perfVariance) / 0.15;

        let agreementMean = this.#agreementScores.reduce((sum, x) => sum + (isValidNumber(x) ? x : 0), 0) / this.#ensembleSize;
        let normalizedAgreement = agreementMean > 0.8 ? -1 : agreementMean < 0.4 ? 1 : (0.6 - agreementMean) / 0.2;

        let delta = 0.01;
        let w_d = 0.4, w_p = 0.3, w_a = 0.3;
        this.#swarmIntelligenceFactor += delta * (
            w_d * (isValidNumber(normalizedDiversity) ? normalizedDiversity : 0) +
            w_p * (isValidNumber(normalizedPerfVariance) ? normalizedPerfVariance : 0) +
            w_a * (isValidNumber(normalizedAgreement) ? normalizedAgreement : 0)
        );
        this.#swarmIntelligenceFactor = Math.max(0.1, Math.min(0.5, this.#swarmIntelligenceFactor));
    }

    #gelu(x) {
        if (!isValidNumber(x)) return 0;
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    }

    #geluDerivative(x) {
        if (!isValidNumber(x)) return 0;

        const clampedX = Math.min(Math.max(x, -10), 10);

        const x3 = Math.pow(clampedX, 3);
        const poly = clampedX + 0.044715 * x3;
        const sqrtTerm = Math.sqrt(2 / Math.PI);
        const tanhArg = sqrtTerm * poly;

        const cdf = 0.5 * (1 + Math.tanh(tanhArg));

        const pdf = Math.exp(-0.5 * clampedX * clampedX) / Math.sqrt(2 * Math.PI);

        const derivative = cdf + clampedX * pdf;

        return isValidNumber(derivative) ? Math.min(Math.max(derivative, -10), 10) : 0;
    }

    #sigmoid (x) {
        return isValidNumber(x) ? 1 / (1 + Math.exp(-Math.min(Math.max(x, -100), 100))) : 0;
    }

    #softmax (arr) {
        if (!arr.every(isValidNumber)) return arr.map(() => 1 / arr.length);
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0) || 1;
        return exp.map(x => x / sum);
    }

    #layerNorm(x, gamma, beta, eps = 1e-6) {
        if (
            !Array.isArray(x) || x.length !== this.#hiddenSize ||
            !Array.isArray(gamma) || gamma.length !== this.#hiddenSize ||
            !Array.isArray(beta) || beta.length !== this.#hiddenSize ||
            !x.every(isValidNumber) || !gamma.every(isValidNumber) || !beta.every(isValidNumber)
        ) {
            return Array(this.#hiddenSize).fill(0);
        }

        const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
        const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / x.length;

        if (!isValidNumber(variance) || !isValidNumber(mean)) {
            return Array(this.#hiddenSize).fill(0);
        }

        const output = x.map((val, i) => {
            const normalized = (val - mean) / Math.sqrt(variance + eps);
            const gammaAdjusted = Math.abs(gamma[i] - 1.0) < 1e-6 ? 1.0 : gamma[i];
            return isValidNumber(normalized) ? gammaAdjusted * normalized + beta[i] : 0;
        });

        return output;
    }

    #contextAwareAttention(inputs, transformerIdx) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber) ||
            !Number.isInteger(transformerIdx) ||
            transformerIdx < 0 ||
            transformerIdx >= this.#ensembleSize
        ) {
            return Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        }

        const transformer = this.#transformers[transformerIdx];

        if (
            !Array.isArray(this.#attentionMemory[transformerIdx]) ||
            this.#attentionMemory[transformerIdx].length === 0 ||
            !this.#attentionMemory[transformerIdx].every(
                seq => Array.isArray(seq) && seq.length === this.#inputSize && seq.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
            )
        ) {
            this.#attentionMemory[transformerIdx] = Array(this.#contextWindow).fill().map(() =>
                Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
            );
        }

        const inputProjection = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j], 0.5), 1.5)
                    : 1;
                inputProjection[i][j] = isValidNumber(inputs[i])
                    ? inputs[i] * specWeight * (1 + this.#swarmIntelligenceFactor)
                    : 0;
            }
        }

        const output = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                output[i][j] = isValidNumber(inputProjection[i][j]) && isValidNumber(transformer.positionalEncoding[i][j])
                    ? inputProjection[i][j] + transformer.positionalEncoding[i][j]
                    : inputProjection[i][j];
            }
        }

        const contextWeight = 0.3;
        if (this.#attentionMemory[transformerIdx].length > 0) {
            const recentAttention = this.#attentionMemory[transformerIdx][this.#attentionMemory[transformerIdx].length - 1];
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#hiddenSize; j++) {
                    const historicalValue = isValidNumber(recentAttention[i][j]) ? recentAttention[i][j] : 0;
                    const specializationFactor = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    output[i][j] = isValidNumber(output[i][j])
                        ? (1 - contextWeight) * output[i][j] + contextWeight * historicalValue * specializationFactor
                        : historicalValue * specializationFactor;
                }
            }
        }

        for (let i = 0; i < this.#inputSize; i++) {
            output[i] = this.#layerNorm(
                output[i],
                transformer.layerNormWeights[0].gamma1,
                transformer.layerNormWeights[0].beta1
            );
        }

        return output;
    }

    #multiHeadAttention(x, layer, transformerIdx) {
        if (
            !Array.isArray(x) ||
            x.length !== this.#inputSize ||
            !x.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !Number.isInteger(transformerIdx) ||
            transformerIdx < 0 ||
            transformerIdx >= this.#ensembleSize ||
            !layer || !layer.Wq || !layer.Wk || !layer.Wv || !layer.Wo ||
            !layer.Wq.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !layer.Wk.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !layer.Wv.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !layer.Wo.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
        ) {
            return {
                output: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Q: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                K: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                V: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                scores: Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                ),
                probs: Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                )
            };
        }

        const headSize = this.#hiddenSize / this.#numHeads;
        if (!Number.isInteger(headSize) || headSize <= 0) {
            return {
                output: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Q: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                K: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                V: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                scores: Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                ),
                probs: Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                )
            };
        }

        const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    Q[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wq[k][j])
                        ? x[i][k] * layer.Wq[k][j] * specWeight
                        : 0;
                    K[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wk[k][j])
                        ? x[i][k] * layer.Wk[k][j] * specWeight
                        : 0;
                    V[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wv[k][j])
                        ? x[i][k] * layer.Wv[k][j] * specWeight
                        : 0;
                }
            }
        }

        const attentionScores = Array(this.#numHeads).fill().map(() =>
            Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
        );
        const attentionProbs = Array(this.#numHeads).fill().map(() =>
            Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
        );
        for (let h = 0; h < this.#numHeads; h++) {
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#inputSize; j++) {
                    let sum = 0;
                    for (let k = 0; k < headSize; k++) {
                        const idx = h * headSize + k;
                        sum += isValidNumber(Q[i][idx]) && isValidNumber(K[j][idx])
                            ? Q[i][idx] * K[j][idx]
                            : 0;
                    }
                    attentionScores[h][i][j] = isValidNumber(sum)
                        ? sum / Math.sqrt(headSize) * this.#attentionScalingFactor * (1 + this.#specializationScores[transformerIdx])
                        : 0;
                }
                attentionProbs[h][i] = this.#softmax(attentionScores[h][i].map(score => isValidNumber(score) ? score : 0));
                attentionProbs[h][i] = this.#dropout(attentionProbs[h][i], this.#dropoutRate, true);
            }
        }

        const output = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let h = 0; h < this.#numHeads; h++) {
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#inputSize; j++) {
                    for (let k = 0; k < headSize; k++) {
                        const idx = h * headSize + k;
                        output[i][idx] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(V[j][idx])
                            ? attentionProbs[h][i][j] * V[j][idx]
                            : 0;
                    }
                }
            }
        }

        const finalOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    finalOutput[i][j] += isValidNumber(output[i][k]) && isValidNumber(layer.Wo[k][j])
                        ? output[i][k] * layer.Wo[k][j] * specWeight
                        : 0;
                }
            }
            finalOutput[i] = this.#dropout(finalOutput[i], this.#dropoutRate, true);
        }

        if (this.#attentionMemory[transformerIdx].length >= this.#contextWindow) {
            this.#attentionMemory[transformerIdx].shift();
        }
        this.#attentionMemory[transformerIdx].push(finalOutput.map(row => row.slice()));

        return {
            output: finalOutput,
            Q,
            K,
            V,
            scores: attentionScores,
            probs: attentionProbs
        };
    }

    #feedForward(x, layer, transformerIdx) {
        if (
            !Array.isArray(x) || x.length !== this.#hiddenSize ||
            !x.every(isValidNumber) ||
            !Number.isInteger(transformerIdx) ||
            transformerIdx < 0 ||
            transformerIdx >= this.#ensembleSize ||
            !layer || !layer.W1 || !layer.W2 || !layer.b1 || !layer.b2 ||
            !layer.W1.every(row => Array.isArray(row) && row.length === this.#feedForwardSize && row.every(isValidNumber)) ||
            !layer.W2.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !Array.isArray(layer.b1) || layer.b1.length !== this.#feedForwardSize || !layer.b1.every(isValidNumber) ||
            !Array.isArray(layer.b2) || layer.b2.length !== this.#hiddenSize || !layer.b2.every(isValidNumber)
        ) {
            return Array(this.#hiddenSize).fill(0);
        }

        const hidden = Array(this.#feedForwardSize).fill(0);
        for (let j = 0; j < this.#feedForwardSize; j++) {
            for (let i = 0; i < this.#hiddenSize; i++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j % this.#hiddenSize])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                    : 1;
                hidden[j] += isValidNumber(x[i]) && isValidNumber(layer.W1[i][j])
                    ? x[i] * layer.W1[i][j] * specWeight
                    : 0;
            }
            hidden[j] = isValidNumber(hidden[j]) && isValidNumber(layer.b1[j])
                ? hidden[j] + layer.b1[j]
                : hidden[j];
        }

        let activated = hidden.map(val => this.#gelu(val));
        activated = this.#dropout(activated, this.#dropoutRate, true);

        const output = Array(this.#hiddenSize).fill(0);
        for (let j = 0; j < this.#hiddenSize; j++) {
            for (let i = 0; i < this.#feedForwardSize; i++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][j % this.#hiddenSize][i % this.#hiddenSize])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][j % this.#hiddenSize][i % this.#hiddenSize], 0.5), 1.5)
                    : 1;
                output[j] += isValidNumber(activated[i]) && isValidNumber(layer.W2[i][j])
                    ? activated[i] * layer.W2[i][j] * specWeight
                    : 0;
            }
            output[j] = isValidNumber(output[j]) && isValidNumber(layer.b2[j])
                ? output[j] + layer.b2[j]
                : output[j];
        }

        return this.#dropout(output, this.#dropoutRate, true);
    }

    #dropout(x, rate, training = false) {
        if (
            !Array.isArray(x) || 
            !x.every(isValidNumber) || 
            !isValidNumber(rate) || 
            rate < 0 || 
            rate >= 1
        ) {
            return x.slice();
        }

        if (!training) return x.slice();

        return x.map(val => {
            if (!isValidNumber(val)) return 0;
            return Math.random() >= rate ? val / (1 - rate) : 0;
        });
    }

    #computeDiversityLoss(outputs) {
        if (
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber) ||
            !this.#trustScoresHistory.every(history => Array.isArray(history) && history.every(isValidNumber))
        ) {
            return 0;
        }

        const meanOutput = outputs.reduce((sum, val) => sum + val, 0) / this.#ensembleSize;

        if (outputs.every(out => Math.abs(out - meanOutput) < 1e-6)) {
            return 0;
        }

        let variance = 0;
        let totalTrustWeight = 0;
        for (let i = 0; i < this.#ensembleSize; i++) {
            const trustWeight = this.#trustScoresHistory[i].length > 0
                ? this.#trustScoresHistory[i][this.#trustScoresHistory[i].length - 1]
                : 0.5;
            if (!isValidNumber(trustWeight) || trustWeight < 0) {
                continue;
            }
            const diff = outputs[i] - meanOutput;
            variance += trustWeight * diff * diff;
            totalTrustWeight += trustWeight;
        }

        variance = totalTrustWeight > 1e-6 ? variance / totalTrustWeight : 0;

        const scaledVariance = variance * 10;
        const diversityLoss = this.#diversityWeight * this.#sigmoid(scaledVariance);

        return isValidNumber(diversityLoss) && diversityLoss >= 0 ? diversityLoss : 0;
    }

    #distillKnowledge(outputs, target) {
        if (
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber) ||
            !isValidNumber(target)
        ) {
            return;
        }

        const sortedIndices = this.#performanceScores
            .map((score, idx) => ({ score: isValidNumber(score) ? score : 0, idx }))
            .sort((a, b) => b.score - a.score);
        const topPerformers = sortedIndices.slice(0, Math.floor(this.#ensembleSize * 0.25)).map(({ idx }) => idx);
        const bottomPerformers = sortedIndices.slice(Math.floor(this.#ensembleSize * 0.25)).map(({ idx }) => idx);

        const topOutputs = topPerformers.map(idx => outputs[idx]);
        const topWeights = topPerformers.map(idx => this.#ensembleWeights[idx]);
        const weightSum = topWeights.reduce((sum, w) => sum + (isValidNumber(w) ? w : 0), 0) || 1;
        const normalizedTopWeights = topWeights.map(w => (isValidNumber(w) ? w : 0) / weightSum);
        let targetOutput = topOutputs.reduce((sum, output, i) =>
            sum + (isValidNumber(output) && isValidNumber(normalizedTopWeights[i]) ? output * normalizedTopWeights[i] : 0), 0
        );
        targetOutput = isValidNumber(targetOutput) ? 0.7 * targetOutput + 0.3 * target : target;

        const diversityLoss = this.#computeDiversityLoss(outputs) * 0.5;

        this.#transformers.forEach((transformer, idx) => {
            if (topPerformers.includes(idx)) return;

            const output = outputs[idx];
            const outputProb = this.#sigmoid(output);
            const targetProb = this.#sigmoid(targetOutput);
            const klLoss = isValidNumber(outputProb) && isValidNumber(targetProb)
                ? outputProb * Math.log((outputProb + 1e-6) / (targetProb + 1e-6)) + (1 - outputProb) * Math.log((1 - outputProb + 1e-6) / (1 - targetProb + 1e-6))
                : 0;

            let l2Reg = 0;
            for (let j = 0; j < this.#hiddenSize; j++) {
                if (isValidNumber(this.#attentionWeightMatrix[idx][j])) {
                    l2Reg += Math.pow(this.#attentionWeightMatrix[idx][j], 2);
                }
            }
            const l2Loss = 0.0005 * l2Reg;
            const totalLoss = 0.5 * klLoss + diversityLoss + l2Loss;

            const adjustedLearningRate = this.#adaptiveLearningRate[idx];
            const grad = isValidNumber(totalLoss) ? totalLoss * adjustedLearningRate : 0;

            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < this.#outputSize; j++) {
                    const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    const gradUpdate = grad * transformer.outputWeights[i][j] * specializationFactor;
                    const clippedUpdate = Math.min(
                        Math.max(adjustedLearningRate * gradUpdate, -this.#gradientClippingThreshold),
                        this.#gradientClippingThreshold
                    );
                    transformer.outputWeights[i][j] = isValidNumber(transformer.outputWeights[i][j])
                        ? transformer.outputWeights[i][j] - clippedUpdate
                        : 0;
                    this.#gradientAccumulation[idx].outputWeights[i][j] += clippedUpdate;
                }
            }
            for (let i = 0; i < this.#outputSize; i++) {
                const gradUpdate = grad;
                const clippedUpdate = Math.min(
                    Math.max(adjustedLearningRate * gradUpdate, -this.#gradientClippingThreshold),
                    this.#gradientClippingThreshold
                );
                transformer.outputBias[i] = isValidNumber(transformer.outputBias[i])
                    ? transformer.outputBias[i] - clippedUpdate
                    : 0;
                this.#gradientAccumulation[idx].outputBias[i] += clippedUpdate;
            }

            for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
                if (
                    !Array.isArray(this.#attentionMemory[idx]) ||
                    this.#attentionMemory[idx].length === 0 ||
                    !this.#attentionMemory[idx].every(
                        seq => Array.isArray(seq) && seq.length === this.#inputSize && seq.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
                    )
                ) {
                    this.#attentionMemory[idx] = Array(this.#contextWindow).fill().map(() =>
                        Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
                    );
                }
                const attentionInput = this.#attentionMemory[idx][this.#attentionMemory[idx].length - 1];
                const headSize = this.#hiddenSize / this.#numHeads;

                const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let i = 0; i < this.#inputSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][k % this.#hiddenSize][j])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][k % this.#hiddenSize][j], 0.5), 1.5)
                                : 1;
                            Q[i][j] += isValidNumber(attentionInput[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wq[k][j])
                                ? attentionInput[i][k] * transformer.attentionWeights[layer].Wq[k][j] * specWeight
                                : 0;
                            K[i][j] += isValidNumber(attentionInput[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wk[k][j])
                                ? attentionInput[i][k] * transformer.attentionWeights[layer].Wk[k][j] * specWeight
                                : 0;
                            V[i][j] += isValidNumber(attentionInput[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wv[k][j])
                                ? attentionInput[i][k] * transformer.attentionWeights[layer].Wv[k][j] * specWeight
                                : 0;
                        }
                    }
                }

                const attentionScores = Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                );
                const attentionProbs = Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                );
                for (let h = 0; h < this.#numHeads; h++) {
                    for (let i = 0; i < this.#inputSize; i++) {
                        for (let j = 0; j < this.#inputSize; j++) {
                            let sum = 0;
                            for (let k = 0; k < headSize; k++) {
                                sum += isValidNumber(Q[i][h * headSize + k]) && isValidNumber(K[j][h * headSize + k])
                                    ? Q[i][h * headSize + k] * K[j][h * headSize + k]
                                    : 0;
                            }
                            attentionScores[h][i][j] = isValidNumber(sum)
                                ? sum / Math.sqrt(headSize) * (1 + this.#specializationScores[idx])
                                : 0;
                        }
                        attentionProbs[h][i] = this.#softmax(attentionScores[h][i].map(score => isValidNumber(score) ? score : 0));
                    }
                }

                const attentionOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let h = 0; h < this.#numHeads; h++) {
                    for (let i = 0; i < this.#inputSize; i++) {
                        for (let j = 0; j < this.#inputSize; j++) {
                            for (let k = 0; k < headSize; k++) {
                                attentionOutput[i][h * headSize + k] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(V[j][h * headSize + k])
                                    ? attentionProbs[h][i][j] * V[j][h * headSize + k]
                                    : 0;
                            }
                        }
                    }
                }

                const woGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const attentionOutputGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(grad));
                for (let i = 0; i < this.#inputSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            woGrad[k][j] += isValidNumber(attentionOutputGrad[i][j]) && isValidNumber(attentionOutput[i][k])
                                ? attentionOutputGrad[i][j] * attentionOutput[i][k]
                                : 0;
                        }
                    }
                }

                const vGrad = Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(headSize).fill(0))
                );
                const scoreGrad = Array(this.#numHeads).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
                );
                for (let h = 0; h < this.#numHeads; h++) {
                    for (let i = 0; i < this.#inputSize; i++) {
                        for (let j = 0; j < this.#inputSize; j++) {
                            for (let k = 0; k < headSize; k++) {
                                const gradIndex = h * headSize + k;
                                scoreGrad[h][i][j] += isValidNumber(attentionOutputGrad[i][gradIndex]) && isValidNumber(V[j][gradIndex])
                                    ? attentionOutputGrad[i][gradIndex] * V[j][gradIndex]
                                    : 0;
                                vGrad[h][j][k] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(attentionOutputGrad[i][gradIndex])
                                    ? attentionProbs[h][i][j] * attentionOutputGrad[i][gradIndex]
                                    : 0;
                            }
                        }
                    }
                }

                const qGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const kGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let h = 0; h < this.#numHeads; h++) {
                    for (let i = 0; i < this.#inputSize; i++) {
                        for (let j = 0; j < this.#inputSize; j++) {
                            for (let k = 0; k < headSize; k++) {
                                const scaledScore = isValidNumber(scoreGrad[h][i][j])
                                    ? scoreGrad[h][i][j] / Math.sqrt(headSize)
                                    : 0;
                                qGrad[i][h * headSize + k] += isValidNumber(scaledScore) && isValidNumber(K[j][h * headSize + k])
                                    ? scaledScore * K[j][h * headSize + k]
                                    : 0;
                                kGrad[j][h * headSize + k] += isValidNumber(scaledScore) && isValidNumber(Q[i][h * headSize + k])
                                    ? scaledScore * Q[i][h * headSize + k]
                                    : 0;
                            }
                        }
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j], 0.5), 1.5)
                            : 1;
                        const wqUpdate = qGrad.reduce((sum, row) => sum + (isValidNumber(row[j]) && isValidNumber(attentionInput[i % this.#inputSize][i]) ? row[j] * attentionInput[i % this.#inputSize][i] : 0), 0);
                        const wkUpdate = kGrad.reduce((sum, row) => sum + (isValidNumber(row[j]) && isValidNumber(attentionInput[i % this.#inputSize][i]) ? row[j] * attentionInput[i % this.#inputSize][i] : 0), 0);
                        const wvUpdate = vGrad.reduce((sum, head) => sum + head[i % this.#inputSize].reduce((s, v) => s + (isValidNumber(v[j % headSize]) ? v[j % headSize] : 0), 0), 0);
                        const woUpdate = isValidNumber(woGrad[i][j]) ? woGrad[i][j] : 0;

                        if (isValidNumber(wqUpdate)) {
                            const clippedUpdate = Math.min(
                                Math.max(adjustedLearningRate * wqUpdate * specializationFactor, -this.#gradientClippingThreshold),
                                this.#gradientClippingThreshold
                            );
                            transformer.attentionWeights[layer].Wq[i][j] = isValidNumber(transformer.attentionWeights[layer].Wq[i][j])
                                ? transformer.attentionWeights[layer].Wq[i][j] - clippedUpdate
                                : 0;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j] += clippedUpdate;
                        }
                        if (isValidNumber(wkUpdate)) {
                            const clippedUpdate = Math.min(
                                Math.max(adjustedLearningRate * wkUpdate * specializationFactor, -this.#gradientClippingThreshold),
                                this.#gradientClippingThreshold
                            );
                            transformer.attentionWeights[layer].Wk[i][j] = isValidNumber(transformer.attentionWeights[layer].Wk[i][j])
                                ? transformer.attentionWeights[layer].Wk[i][j] - clippedUpdate
                                : 0;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j] += clippedUpdate;
                        }
                        if (isValidNumber(wvUpdate)) {
                            const clippedUpdate = Math.min(
                                Math.max(adjustedLearningRate * wvUpdate * specializationFactor, -this.#gradientClippingThreshold),
                                this.#gradientClippingThreshold
                            );
                            transformer.attentionWeights[layer].Wv[i][j] = isValidNumber(transformer.attentionWeights[layer].Wv[i][j])
                                ? transformer.attentionWeights[layer].Wv[i][j] - clippedUpdate
                                : 0;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j] += clippedUpdate;
                        }
                        if (isValidNumber(woUpdate)) {
                            const clippedUpdate = Math.min(
                                Math.max(adjustedLearningRate * woUpdate * specializationFactor, -this.#gradientClippingThreshold),
                                this.#gradientClippingThreshold
                            );
                            transformer.attentionWeights[layer].Wo[i][j] = isValidNumber(transformer.attentionWeights[layer].Wo[i][j])
                                ? transformer.attentionWeights[layer].Wo[i][j] - clippedUpdate
                                : 0;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j] += clippedUpdate;
                        }
                    }
                }

                const ffnInput = attentionInput[0] || Array(this.#hiddenSize).fill(0);
                const hidden = Array(this.#feedForwardSize).fill(0);
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        hidden[j] += isValidNumber(ffnInput[i]) && isValidNumber(transformer.ffnWeights[layer].W1[i][j])
                            ? ffnInput[i] * transformer.ffnWeights[layer].W1[i][j]
                            : 0;
                    }
                }
                const activated = hidden.map((val, i) => this.#gelu(val + (isValidNumber(transformer.ffnWeights[layer].b1[i]) ? transformer.ffnWeights[layer].b1[i] : 0)));
                let ffnGrad = Array(this.#feedForwardSize).fill(0);
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        ffnGrad[i] += isValidNumber(grad) && isValidNumber(transformer.ffnWeights[layer].W2[i][j])
                            ? grad * transformer.ffnWeights[layer].W2[i][j]
                            : 0;
                    }
                    ffnGrad[i] = isValidNumber(ffnGrad[i]) ? ffnGrad[i] * this.#geluDerivative(hidden[i]) : 0;
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                            : 1;
                        const update = adjustedLearningRate * ffnGrad[j] * ffnInput[i] * specializationFactor;
                        if (isValidNumber(update)) {
                            const clippedUpdate = Math.min(
                                Math.max(update, -this.#gradientClippingThreshold),
                                this.#gradientClippingThreshold
                            );
                            transformer.ffnWeights[layer].W1[i][j] = isValidNumber(transformer.ffnWeights[layer].W1[i][j])
                                ? transformer.ffnWeights[layer].W1[i][j] - clippedUpdate
                                : 0;
                            this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] += clippedUpdate;
                        }
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    const update = adjustedLearningRate * ffnGrad[i];
                    if (isValidNumber(update)) {
                        const clippedUpdate = Math.min(
                            Math.max(update, -this.#gradientClippingThreshold),
                            this.#gradientClippingThreshold
                        );
                        transformer.ffnWeights[layer].b1[i] = isValidNumber(transformer.ffnWeights[layer].b1[i])
                            ? transformer.ffnWeights[layer].b1[i] - clippedUpdate
                            : 0;
                        this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] += clippedUpdate;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize], 0.5), 1.5)
                            : 1;
                        const update = adjustedLearningRate * grad * activated[i] * specializationFactor;
                        if (isValidNumber(update)) {
                            const clippedUpdate = Math.min(
                                Math.max(update, -this.#gradientClippingThreshold),
                                this.#gradientClippingThreshold
                            );
                            transformer.ffnWeights[layer].W2[i][j] = isValidNumber(transformer.ffnWeights[layer].W2[i][j])
                                ? transformer.ffnWeights[layer].W2[i][j] - clippedUpdate
                                : 0;
                            this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] += clippedUpdate;
                        }
                    }
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    const update = adjustedLearningRate * grad;
                    if (isValidNumber(update)) {
                        const clippedUpdate = Math.min(
                            Math.max(update, -this.#gradientClippingThreshold),
                            this.#gradientClippingThreshold
                        );
                        transformer.ffnWeights[layer].b2[i] = isValidNumber(transformer.ffnWeights[layer].b2[i])
                            ? transformer.ffnWeights[layer].b2[i] - clippedUpdate
                            : 0;
                        this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] += clippedUpdate;
                    }
                }

                const normInput = attentionInput;
                for (let i = 0; i < this.#inputSize; i++) {
                    const gamma1Grad = Array(this.#hiddenSize).fill(0);
                    const beta1Grad = Array(this.#hiddenSize).fill(0);
                    const gamma2Grad = Array(this.#hiddenSize).fill(0);
                    const beta2Grad = Array(this.#hiddenSize).fill(0);
                    const meanNorm = normInput[i].reduce((sum, val) => sum + val, 0) / this.#hiddenSize;
                    const varianceNorm = normInput[i].reduce((sum, val) => sum + Math.pow(val - meanNorm, 2), 0) / this.#hiddenSize;
                    const stdNorm = Math.sqrt(varianceNorm + 1e-6);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const normalized = (normInput[i][j] - meanNorm) / stdNorm;
                        gamma1Grad[j] = isValidNumber(qGrad[i][j]) && isValidNumber(normalized)
                            ? qGrad[i][j] * normalized
                            : 0;
                        beta1Grad[j] = isValidNumber(qGrad[i][j]) ? qGrad[i][j] : 0;
                        gamma2Grad[j] = isValidNumber(grad) && isValidNumber(normalized)
                            ? grad * normalized
                            : 0;
                        beta2Grad[j] = isValidNumber(grad) ? grad : 0;
                        const updateGamma1 = adjustedLearningRate * gamma1Grad[j];
                        const updateBeta1 = adjustedLearningRate * beta1Grad[j];
                        const updateGamma2 = adjustedLearningRate * gamma2Grad[j];
                        const updateBeta2 = adjustedLearningRate * beta2Grad[j];
                        if (isValidNumber(updateGamma1)) {
                            const clippedUpdate = Math.min(
                                Math.max(updateGamma1, -this.#gradientClippingThreshold * 1.5),
                                this.#gradientClippingThreshold * 1.5
                            );
                            transformer.layerNormWeights[layer].gamma1[j] = isValidNumber(transformer.layerNormWeights[layer].gamma1[j])
                                ? transformer.layerNormWeights[layer].gamma1[j] - clippedUpdate
                                : 1;
                            this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[j] += clippedUpdate;
                        }
                        if (isValidNumber(updateBeta1)) {
                            const clippedUpdate = Math.min(
                                Math.max(updateBeta1, -this.#gradientClippingThreshold),
                                this.#gradientClippingThreshold
                            );
                            transformer.layerNormWeights[layer].beta1[j] = isValidNumber(transformer.layerNormWeights[layer].beta1[j])
                                ? transformer.layerNormWeights[layer].beta1[j] - clippedUpdate
                                : 0;
                            this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[j] += clippedUpdate;
                        }
                        if (isValidNumber(updateGamma2)) {
                            const clippedUpdate = Math.min(
                                Math.max(updateGamma2, -this.#gradientClippingThreshold * 1.5),
                                this.#gradientClippingThreshold * 1.5
                            );
                            transformer.layerNormWeights[layer].gamma2[j] = isValidNumber(transformer.layerNormWeights[layer].gamma2[j])
                                ? transformer.layerNormWeights[layer].gamma2[j] - clippedUpdate
                                : 1;
                            this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[j] += clippedUpdate;
                        }
                        if (isValidNumber(updateBeta2)) {
                            const clippedUpdate = Math.min(
                                Math.max(updateBeta2, -this.#gradientClippingThreshold),
                                this.#gradientClippingThreshold
                            );
                            transformer.layerNormWeights[layer].beta2[j] = isValidNumber(transformer.layerNormWeights[layer].beta2[j])
                                ? transformer.layerNormWeights[layer].beta2[j] - clippedUpdate
                                : 0;
                            this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[j] += clippedUpdate;
                        }
                    }
                }
            }});

        this.#gradientAccumulation = this.#gradientAccumulation.map(() => ({
            outputWeights: Array(this.#hiddenSize).fill().map(() => Array(this.#outputSize).fill(0)),
            outputBias: Array(this.#outputSize).fill(0),
            attentionWeights: Array(this.#numLayers).fill().map(() => ({
                Wq: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wk: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wv: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                Wo: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
            })),
            ffnWeights: Array(this.#numLayers).fill().map(() => ({
                W1: Array(this.#hiddenSize).fill().map(() => Array(this.#feedForwardSize).fill(0)),
                W2: Array(this.#feedForwardSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
                b1: Array(this.#feedForwardSize).fill(0),
                b2: Array(this.#hiddenSize).fill(0)
            })),
            layerNormWeights: Array(this.#numLayers).fill().map(() => ({
                gamma1: Array(this.#hiddenSize).fill(0),
                beta1: Array(this.#hiddenSize).fill(0),
                gamma2: Array(this.#hiddenSize).fill(0),
                beta2: Array(this.#hiddenSize).fill(0)
            })),
            attentionBias: Array(this.#hiddenSize).fill(0)
        }));
    }

    forward(inputs) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber)
        ) {
            return Array(this.#outputSize).fill(0);
        }

        const outputs = this.#transformers.map((transformer, idx) => {
            if (
                !Array.isArray(this.#attentionMemory[idx]) ||
                this.#attentionMemory[idx].length === 0 ||
                !this.#attentionMemory[idx][0].every(row => Array.isArray(row) && row.length === this.#hiddenSize)
            ) {
                this.#attentionMemory[idx] = Array(this.#contextWindow).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
                );
            }

            let x = this.#contextAwareAttention(inputs, idx);

            for (let layer = 0; layer < this.#numLayers; layer++) {
                const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));

                const attentionResult = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx);
                const attentionOutput = attentionResult.output;

                const attentionResidual = x.map((row, i) => row.map((val, j) =>
                    isValidNumber(val) && isValidNumber(attentionOutput[i][j]) ? val + attentionOutput[i][j] : val
                ));

                const normAttention = attentionResidual.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));

                const ffnOutput = this.#feedForward(normAttention[0], transformer.ffnWeights[layer]);

                x = attentionResidual.map((row, i) => row.map((val, j) =>
                    isValidNumber(val) && isValidNumber(ffnOutput[j]) ? val + ffnOutput[j] : val
                ));
            }

            let output = Array(this.#outputSize).fill(0);
            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < this.#outputSize; j++) {
                    output[j] += isValidNumber(x[0][i]) && isValidNumber(transformer.outputWeights[i][j])
                        ? x[0][i] * transformer.outputWeights[i][j]
                        : 0;
                }
            }

            output = output.map((val, i) => isValidNumber(val) && isValidNumber(transformer.outputBias[i])
                ? val + transformer.outputBias[i]
                : val
            );
            return output[0];
        });

        const ensembleWeights = this.#computeAttentionWeights(inputs, outputs);
        this.#ensembleWeights = ensembleWeights;
        this.#normalizeEnsembleWeights();

        const finalOutput = Array(this.#outputSize).fill(0);
        for (let i = 0; i < this.#ensembleSize; i++) {
            if (isValidNumber(outputs[i]) && isValidNumber(this.#ensembleWeights[i])) {
                finalOutput[0] += outputs[i] * this.#ensembleWeights[i];
            }
        }

        return [this.#sigmoid(isValidNumber(finalOutput[0]) ? finalOutput[0] : 0)];
    }

    train(inputs, target, winRate = 0.5, shouldSave = true) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber) ||
            !isValidNumber(target) ||
            !isValidNumber(winRate) ||
            winRate < 0 ||
            winRate > 1
        ) {
            return;
        }
        this.#trainingStepCount++;

        const individualOutputs = [];
        const layerOutputs = this.#transformers.map(() => []);
        const activations = this.#transformers.map(() => []);
        const attentionIntermediates = this.#transformers.map(() => []);

        this.#transformers.forEach((transformer, idx) => {
            let x = this.#contextAwareAttention(inputs, idx);
            layerOutputs[idx].push(x);
            const transformerActivations = [];
            const transformerAttentionIntermediates = [];

            for (let layer = 0; layer < this.#numLayers; layer++) {
                const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));
                const attentionResult = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx);
                const attentionOutput = attentionResult.output;
                const attentionResidual = x.map((row, i) => row.map((val, j) =>
                    isValidNumber(val) && isValidNumber(attentionOutput[i][j]) ? val + attentionOutput[i][j] : val
                ));
                const normAttention = attentionResidual.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));
                const ffnOutput = this.#feedForward(normAttention[0], transformer.ffnWeights[layer]);
                x = attentionResidual.map((row, i) => row.map((val, j) =>
                    isValidNumber(val) && isValidNumber(ffnOutput[j]) ? val + ffnOutput[j] : val
                ));
                layerOutputs[idx].push(x);
                transformerActivations.push({ normX, attentionOutput, normAttention });
                transformerAttentionIntermediates.push({
                    Q: attentionResult.Q,
                    K: attentionResult.K,
                    V: attentionResult.V,
                    attentionScores: attentionResult.scores,
                    attentionProbs: attentionResult.probs
                });
            }
            activations[idx] = transformerActivations;
            attentionIntermediates[idx] = transformerAttentionIntermediates;

            let output = Array(this.#outputSize).fill(0);
            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < this.#outputSize; j++) {
                    output[j] += isValidNumber(x[0][i]) && isValidNumber(transformer.outputWeights[i][j])
                        ? x[0][i] * transformer.outputWeights[i][j]
                        : 0;
                }
            }
            output = output.map((val, i) => isValidNumber(val) && isValidNumber(transformer.outputBias[i])
                ? val + transformer.outputBias[i]
                : val
            );
            individualOutputs[idx] = this.#sigmoid(output[0]);
        });

        const ensembleWeights = this.#computeAttentionWeights(inputs, individualOutputs);
        this.#ensembleWeights = ensembleWeights;
        this.#normalizeEnsembleWeights();
        const output = individualOutputs.reduce((sum, out, idx) =>
            sum + (isValidNumber(out) && isValidNumber(this.#ensembleWeights[idx]) ? out * this.#ensembleWeights[idx] : 0), 0
        );
        const error = target - output;

        const dL_d_output = error;
        const dL_d_w = individualOutputs.map(out => dL_d_output * out);

        const dL_d_scores = Array(this.#ensembleSize).fill(0);
        for (let t = 0; t < this.#ensembleSize; t++) {
            for (let s = 0; s < this.#ensembleSize; s++) {
                if (s === t) {
                    dL_d_scores[t] += dL_d_w[s] * ensembleWeights[s] * (1 - ensembleWeights[s]);
                } else {
                    dL_d_scores[t] -= dL_d_w[s] * ensembleWeights[s] * ensembleWeights[t];
                }
            }
        }

        for (let t = 0; t < this.#ensembleSize; t++) {
            const grad = dL_d_scores[t] / this.#hiddenSize;
            for (let k = 0; k < this.#hiddenSize; k++) {
                this.#gradientAccumulation[t].attentionBias[k] += grad;
            }
        }

        this.#performanceScores = this.#performanceScores.map((score, idx) => {
            const individualError = Math.abs(target - individualOutputs[idx]);
            const newScore = 0.9 * score + 0.1 * (1 - individualError);
            this.#historicalPerformance[idx].push(isValidNumber(newScore) ? newScore : 0);
            if (this.#historicalPerformance[idx].length > this.#maxPerformanceHistory) {
                this.#historicalPerformance[idx].shift();
            }
            return isValidNumber(newScore) ? newScore : 0;
        });

        this.#agreementScores = this.#agreementScores.map((score, idx) => {
            const agreement = 1 - Math.abs(individualOutputs[idx] - output);
            const newScore = 0.9 * score + 0.1 * (isValidNumber(agreement) ? agreement : 0);
            return isValidNumber(newScore) ? newScore : 0;
        });

        this.#updateTrustScores();
        this.#computeSpecializationScores(inputs, individualOutputs);
        this.#updateAdaptiveLearningRates();
        this.#adjustSwarmFactor(individualOutputs);

        this.#transformers.forEach((transformer, idx) => {
            const adjustedLearningRate = this.#adaptiveLearningRate[idx] * (0.5 + 0.5 * winRate);
            const delta = Math.min(Math.max(error * output * (1 - output) * adjustedLearningRate, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
            let grad = Array(this.#hiddenSize).fill(0);

            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < this.#outputSize; j++) {
                    const gradUpdate = isValidNumber(delta) && isValidNumber(this.#ensembleWeights[idx]) && isValidNumber(layerOutputs[idx][this.#numLayers][0][i])
                        ? delta * this.#ensembleWeights[idx] * layerOutputs[idx][this.#numLayers][0][i]
                        : 0;
                    const clippedUpdate = Math.min(Math.max(gradUpdate, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
                    this.#gradientAccumulation[idx].outputWeights[i][j] += clippedUpdate;
                    grad[i] += isValidNumber(delta) && isValidNumber(transformer.outputWeights[i][j])
                        ? delta * transformer.outputWeights[i][j]
                        : 0;
                }
            }
            this.#gradientAccumulation[idx].outputBias[0] += isValidNumber(delta) ? delta : 0;

            for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
                const x = layerOutputs[idx][layer];
                const { normX, attentionOutput, normAttention } = activations[idx][layer];
                const { Q, K, V, attentionScores, attentionProbs } = attentionIntermediates[idx][layer];
                const headSize = this.#hiddenSize / this.#numHeads;

                const ffnInput = normAttention[0];
                const hidden = Array(this.#feedForwardSize).fill(0);
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        hidden[j] += isValidNumber(ffnInput[i]) && isValidNumber(transformer.ffnWeights[layer].W1[i][j])
                            ? ffnInput[i] * transformer.ffnWeights[layer].W1[i][j]
                            : 0;
                    }
                }
                const activated = hidden.map((val, i) => this.#gelu(val + (isValidNumber(transformer.ffnWeights[layer].b1[i]) ? transformer.ffnWeights[layer].b1[i] : 0)));
                let ffnGrad = Array(this.#feedForwardSize).fill(0);
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        ffnGrad[i] += isValidNumber(grad[j]) && isValidNumber(transformer.ffnWeights[layer].W2[i][j])
                            ? grad[j] * transformer.ffnWeights[layer].W2[i][j]
                            : 0;
                    }
                    ffnGrad[i] = isValidNumber(ffnGrad[i]) ? ffnGrad[i] * this.#geluDerivative(hidden[i]) : 0;
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        const update = adjustedLearningRate * ffnGrad[j] * ffnInput[i];
                        this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] += isValidNumber(update)
                            ? Math.min(Math.max(update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                            : 0;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    const update = adjustedLearningRate * ffnGrad[i];
                    this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] += isValidNumber(update)
                        ? Math.min(Math.max(update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                        : 0;
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const update = adjustedLearningRate * grad[j] * activated[i];
                        this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] += isValidNumber(update)
                            ? Math.min(Math.max(update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                            : 0;
                    }
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    const update = adjustedLearningRate * grad[i];
                    this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] += isValidNumber(update)
                        ? Math.min(Math.max(update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                        : 0;
                }

                const normAttentionGrad = grad.slice();
                const gamma2Grad = Array(this.#hiddenSize).fill(0);
                const beta2Grad = Array(this.#hiddenSize).fill(0);
                const meanAttention = normAttention[0].reduce((sum, val) => sum + val, 0) / this.#hiddenSize;
                const varianceAttention = normAttention[0].reduce((sum, val) => sum + Math.pow(val - meanAttention, 2), 0) / this.#hiddenSize;
                const stdAttention = Math.sqrt(varianceAttention + 1e-6);
                for (let i = 0; i < this.#hiddenSize; i++) {
                    const normalized = (normAttention[0][i] - meanAttention) / stdAttention;
                    gamma2Grad[i] = isValidNumber(normAttentionGrad[i]) && isValidNumber(normalized)
                        ? normAttentionGrad[i] * normalized
                        : 0;
                    beta2Grad[i] = isValidNumber(normAttentionGrad[i]) ? normAttentionGrad[i] : 0;
                    const updateGamma = adjustedLearningRate * gamma2Grad[i];
                    const updateBeta = adjustedLearningRate * beta2Grad[i];
                    this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i] += isValidNumber(updateGamma)
                        ? Math.min(Math.max(updateGamma, -this.#gradientClippingThreshold * 1.5), this.#gradientClippingThreshold * 1.5)
                        : 0;
                    this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i] += isValidNumber(updateBeta)
                        ? Math.min(Math.max(updateBeta, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                        : 0;
                }

                const attentionGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let i = 0; i < this.#inputSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        attentionGrad[i][j] = grad[j];
                    }
                }
                const woGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let i = 0; i < this.#inputSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            woGrad[k][j] += isValidNumber(attentionGrad[i][j]) && isValidNumber(attentionOutput[i][k])
                                ? attentionGrad[i][j] * attentionOutput[i][k]
                                : 0;
                        }
                    }
                }
                const vGrad = Array(this.#numHeads).fill().map(() => Array(this.#inputSize).fill().map(() => Array(headSize).fill(0)));
                const scoreGrad = Array(this.#numHeads).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0)));
                for (let h = 0; h < this.#numHeads; h++) {
                    for (let i = 0; i < this.#inputSize; i++) {
                        for (let j = 0; j < this.#inputSize; j++) {
                            for (let k = 0; k < headSize; k++) {
                                const idx = h * headSize + k;
                                vGrad[h][j][k] += isValidNumber(attentionGrad[i][idx]) && isValidNumber(attentionProbs[h][i][j])
                                    ? attentionGrad[i][idx] * attentionProbs[h][i][j]
                                    : 0;
                                scoreGrad[h][i][j] += isValidNumber(attentionGrad[i][idx]) && isValidNumber(V[j][idx])
                                    ? attentionGrad[i][idx] * V[j][idx]
                                    : 0;
                            }
                        }
                    }
                }
                const qGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const kGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let h = 0; h < this.#numHeads; h++) {
                    for (let i = 0; i < this.#inputSize; i++) {
                        for (let j = 0; j < this.#inputSize; j++) {
                            for (let k = 0; k < headSize; k++) {
                                const idx = h * headSize + k;
                                const scaledScore = isValidNumber(scoreGrad[h][i][j])
                                    ? scoreGrad[h][i][j] / Math.sqrt(headSize)
                                    : 0;
                                qGrad[i][idx] += isValidNumber(scaledScore) && isValidNumber(K[j][idx])
                                    ? scaledScore * K[j][idx]
                                    : 0;
                                kGrad[j][idx] += isValidNumber(scaledScore) && isValidNumber(Q[i][idx])
                                    ? scaledScore * Q[i][idx]
                                    : 0;
                            }
                        }
                    }
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const wqUpdate = qGrad.reduce((sum, row) => sum + (isValidNumber(row[j]) && isValidNumber(normX[i % this.#inputSize][i]) ? row[j] * normX[i % this.#inputSize][i] : 0), 0);
                        const wkUpdate = kGrad.reduce((sum, row) => sum + ( isValidNumber(row[j]) && isValidNumber(normX[i % this.#inputSize][i]) ? row[j] * normX[i % this.#inputSize][i] : 0), 0);
                        const wvUpdate = vGrad.reduce((sum, head) => sum + head[i % this.#inputSize].reduce((s, v) => s + (isValidNumber(v[j % headSize]) ? v[j % headSize] : 0), 0), 0);
                        const woUpdate = isValidNumber(woGrad[i][j]) ? woGrad[i][j] : 0;
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                            ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j]
                            : 1;
                        if (isValidNumber(wqUpdate)) {
                            const clippedUpdate = Math.min(Math.max(adjustedLearningRate * wqUpdate * specializationFactor, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j] += clippedUpdate;
                        }
                        if (isValidNumber(wkUpdate)) {
                            const clippedUpdate = Math.min(Math.max(adjustedLearningRate * wkUpdate * specializationFactor, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j] += clippedUpdate;
                        }
                        if (isValidNumber(wvUpdate)) {
                            const clippedUpdate = Math.min(Math.max(adjustedLearningRate * wvUpdate * specializationFactor, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j] += clippedUpdate;
                        }
                        if (isValidNumber(woUpdate)) {
                            const clippedUpdate = Math.min(Math.max(adjustedLearningRate * woUpdate * specializationFactor, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j] += clippedUpdate;
                        }
                    }
                }

                const normXGrad = qGrad;
                for (let i = 0; i < this.#inputSize; i++) {
                    const gamma1Grad = Array(this.#hiddenSize).fill(0);
                    const beta1Grad = Array(this.#hiddenSize).fill(0);
                    const meanX = normX[i].reduce((sum, val) => sum + val, 0) / this.#hiddenSize;
                    const varianceX = normX[i].reduce((sum, val) => sum + Math.pow(val - meanX, 2), 0) / this.#hiddenSize;
                    const stdX = Math.sqrt(varianceX + 1e-6);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const normalized = (normX[i][j] - meanX) / stdX;
                        gamma1Grad[j] = isValidNumber(normXGrad[i][j]) && isValidNumber(normalized)
                            ? normXGrad[i][j] * normalized
                            : 0;
                        beta1Grad[j] = isValidNumber(normXGrad[i][j]) ? normXGrad[i][j] : 0;
                        const updateGamma = adjustedLearningRate * gamma1Grad[j];
                        const updateBeta = adjustedLearningRate * beta1Grad[j];
                        this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[j] += isValidNumber(updateGamma)
                            ? Math.min(Math.max(updateGamma, -this.#gradientClippingThreshold * 1.5), this.#gradientClippingThreshold * 1.5)
                            : 0;
                        this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[j] += isValidNumber(updateBeta)
                            ? Math.min(Math.max(updateBeta, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                            : 0;
                    }
                }
                grad = qGrad[0];
            }
        });

        this.#transformers.forEach((transformer, idx) => {
            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < this.#outputSize; j++) {
                    transformer.outputWeights[i][j] -= isValidNumber(this.#gradientAccumulation[idx].outputWeights[i][j])
                        ? this.#gradientAccumulation[idx].outputWeights[i][j]
                        : 0;
                }
            }
            for (let j = 0; j < this.#outputSize; j++) {
                transformer.outputBias[j] -= isValidNumber(this.#gradientAccumulation[idx].outputBias[j])
                    ? this.#gradientAccumulation[idx].outputBias[j]
                    : 0;
            }
            for (let layer = 0; layer < this.#numLayers; layer++) {
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        transformer.ffnWeights[layer].W1[i][j] -= isValidNumber(this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j])
                            ? this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j]
                            : 0;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        transformer.ffnWeights[layer].W2[i][j] -= isValidNumber(this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j])
                            ? this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j]
                            : 0;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    transformer.ffnWeights[layer].b1[i] -= isValidNumber(this.#gradientAccumulation[idx].ffnWeights[layer].b1[i])
                        ? this.#gradientAccumulation[idx].ffnWeights[layer].b1[i]
                        : 0;
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    transformer.ffnWeights[layer].b2[i] -= isValidNumber(this.#gradientAccumulation[idx].ffnWeights[layer].b2[i])
                        ? this.#gradientAccumulation[idx].ffnWeights[layer].b2[i]
                        : 0;
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        transformer.attentionWeights[layer].Wq[i][j] -= isValidNumber(this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j])
                            ? this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j]
                            : 0;
                        transformer.attentionWeights[layer].Wk[i][j] -= isValidNumber(this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j])
                            ? this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j]
                            : 0;
                        transformer.attentionWeights[layer].Wv[i][j] -= isValidNumber(this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j])
                            ? this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j]
                            : 0;
                        transformer.attentionWeights[layer].Wo[i][j] -= isValidNumber(this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j])
                            ? this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j]
                            : 0;
                    }
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    transformer.layerNormWeights[layer].gamma1[i] -= isValidNumber(this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i])
                        ? this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i]
                        : 0;
                    transformer.layerNormWeights[layer].beta1[i] -= isValidNumber(this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[i])
                        ? this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[i]
                        : 0;
                    transformer.layerNormWeights[layer].gamma2[i] -= isValidNumber(this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i])
                        ? this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i]
                        : 0;
                    transformer.layerNormWeights[layer].beta2[i] -= isValidNumber(this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i])
                        ? this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i]
                        : 0;
                }
            }

            for (let i = 0; i < this.#hiddenSize; i++) {
                const l2PenaltyBias = 0.001 * this.#attentionBias[idx][i];
                const biasUpdate = isValidNumber(this.#gradientAccumulation[idx].attentionBias[i])
                    ? this.#gradientAccumulation[idx].attentionBias[i] + l2PenaltyBias
                    : l2PenaltyBias;
                this.#attentionBias[idx][i] -= Math.min(Math.max(biasUpdate, -0.1), 0.1);

                const l2PenaltyWeight = 0.001 * this.#attentionWeightMatrix[idx][i];
                const weightUpdate = isValidNumber(this.#gradientAccumulation[idx].attentionBias[i])
                    ? this.#gradientAccumulation[idx].attentionBias[i] + l2PenaltyWeight
                    : l2PenaltyWeight;
                this.#attentionWeightMatrix[idx][i] -= Math.min(Math.max(weightUpdate, -0.1), 0.1);
            }
        });

        this.#distillKnowledge(individualOutputs, target);
        if (this.#shouldCommunicate()) {
            this.#shareWeights();
        }
        this.#normalizeAttentionWeights();

        if (shouldSave) {
            this.#saveState();
        }
    }

    dumpState() {
        this.#saveState()
    }
}

export default HiveMind;