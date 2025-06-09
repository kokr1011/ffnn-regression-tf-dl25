function setup() {
  createCanvas(400, 400);
}

function draw() {
  background(220);
}

 // --- Daten und Rauschen ---
    function targetFunction(x) {
      return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
    }

    function gaussianNoise(mean = 0, variance = 0.05) {
      const u1 = Math.random();
      const u2 = Math.random();
      const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      return mean + Math.sqrt(variance) * randStdNormal;
    }

    function generateDataset(N = 100, addNoise = false, variance = 0.05) {
      const x = [];
      const y = [];
      for (let i = 0; i < N; i++) {
        const xi = -2 + 4 * Math.random();
        const yi = targetFunction(xi);
        x.push(xi);
        y.push(addNoise ? yi + gaussianNoise(0, variance) : yi);
      }
      return { x, y };
    }

    function splitDataset(x, y) {
      const indices = [...Array(x.length).keys()].sort(() => Math.random() - 0.5);
      const split = Math.floor(x.length / 2);
      const train = { x: [], y: [] };
      const test = { x: [], y: [] };

      for (let i = 0; i < x.length; i++) {
        const idx = indices[i];
        if (i < split) {
          train.x.push(x[idx]);
          train.y.push(y[idx]);
        } else {
          test.x.push(x[idx]);
          test.y.push(y[idx]);
        }
      }
      return { train, test };
    }

    // Erste Ausführung
    const cleanData = generateDataset(100, false);
    const noisyData = generateDataset(100, true);
    const cleanSplit = splitDataset(cleanData.x, cleanData.y);
    const noisySplit = splitDataset(noisyData.x, noisyData.y);

    console.log('Clean Data:', cleanSplit);
    console.log('Noisy Data:', noisySplit);

  // --- Modellaufbau und Training ---
function createFFNNModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    return model;
  }

 async function trainModel(model, xTrain, yTrain, epochs = 100) {
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'meanSquaredError'
    });

    const xs = tf.tensor2d(xTrain, [xTrain.length, 1]);
    const ys = tf.tensor2d(yTrain, [yTrain.length, 1]);

    const history = await model.fit(xs, ys, {
      batchSize: 32,
      epochs: epochs,
      shuffle: true
    });

    xs.dispose();
    ys.dispose();
    return history;
  }

// --- Vorhersagefunktion und Loss-Berechnung ---
  async function evaluateModel(model, xData, yData) {
    const xs = tf.tensor2d(xData, [xData.length, 1]);
    const ys = tf.tensor2d(yData, [yData.length, 1]);
    const preds = model.predict(xs);
    const loss = await model.evaluate(xs, ys).data();

    const predictions = await preds.array();
    xs.dispose();
    ys.dispose();
    preds.dispose();
    return { predictions: predictions.map(p => p[0]), loss: loss[0] };
  }

  // --- Visualisierung mit Chart.js ---
function plotRawData(canvasId, trainSet, testSet, title) {
  const trainData = trainSet.x.map((x, i) => ({ x, y: trainSet.y[i] }));
  const testData = testSet.x.map((x, i) => ({ x, y: testSet.y[i] }));

  new Chart(document.getElementById(canvasId), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Trainingsdaten',
          data: trainData,
          backgroundColor: 'green'
        },
        {
          label: 'Testdaten',
          data: testData,
          backgroundColor: 'orange'
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: title
        }
      },
      scales: {
        x: {
          type: 'linear',
          position: 'bottom'
        }
      }
    }
  });
}

// direkt nach Datengenerierung und Aufteilung aufrufen:
plotRawData('dataset-clean', cleanSplit.train, cleanSplit.test, 'Datensatz ohne Rauschen');
plotRawData('dataset-noisy', noisySplit.train, noisySplit.test, 'Datensatz mit Rauschen');




    function plotPrediction(canvasId, xData, yTrue, yPred, title) {
    const truePoints = xData.map((x, i) => ({ x, y: yTrue[i] }));
    const predPoints = xData.map((x, i) => ({ x, y: yPred[i] }));

    new Chart(document.getElementById(canvasId), {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Wahr',
            data: truePoints,
            backgroundColor: 'blue'
          },
          {
            label: 'Vorhersage',
            data: predPoints,
            backgroundColor: 'red'
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: title
          }
        },
        scales: {
          x: {
            type: 'linear',
            position: 'bottom'
          }
        }
      }
    });
  }
  // --- Beispielmodell trainieren und anzeigen ---
  async function runDemo() {
    const model = createFFNNModel();
    await trainModel(model, cleanSplit.train.x, cleanSplit.train.y, 100);

    const trainEval = await evaluateModel(model, cleanSplit.train.x, cleanSplit.train.y);
    const testEval = await evaluateModel(model, cleanSplit.test.x, cleanSplit.test.y);

    document.getElementById('loss-clean-train').innerText = trainEval.loss.toFixed(4);
    document.getElementById('loss-clean-test').innerText = testEval.loss.toFixed(4);

    plotPrediction('pred-clean-train', cleanSplit.train.x, cleanSplit.train.y, trainEval.predictions, 'Trainingsdaten – ohne Rauschen');
    plotPrediction('pred-clean-test', cleanSplit.test.x, cleanSplit.test.y, testEval.predictions, 'Testdaten – ohne Rauschen');
  }

  runDemo();


async function runBestFitModel() {
    const bestModel = createFFNNModel();
    await trainModel(bestModel, noisySplit.train.x, noisySplit.train.y, 100);

    const bestTrainEval = await evaluateModel(bestModel, noisySplit.train.x, noisySplit.train.y);
    const bestTestEval = await evaluateModel(bestModel, noisySplit.test.x, noisySplit.test.y);

    document.getElementById('loss-best-train').innerText = bestTrainEval.loss.toFixed(4);
    document.getElementById('loss-best-test').innerText = bestTestEval.loss.toFixed(4);

    plotPrediction('pred-best-train', noisySplit.train.x, noisySplit.train.y, bestTrainEval.predictions, 'Trainingsdaten – mit Rauschen');
    plotPrediction('pred-best-test', noisySplit.test.x, noisySplit.test.y, bestTestEval.predictions, 'Testdaten – mit Rauschen');
  }

  runBestFitModel();

 async function runOverfitModel() {
    const overfitModel = createFFNNModel();
    await trainModel(overfitModel, noisySplit.train.x, noisySplit.train.y, 1000); // viele Epochen für Overfitting

    const overfitTrainEval = await evaluateModel(overfitModel, noisySplit.train.x, noisySplit.train.y);
    const overfitTestEval = await evaluateModel(overfitModel, noisySplit.test.x, noisySplit.test.y);

    document.getElementById('loss-overfit-train').innerText = overfitTrainEval.loss.toFixed(4);
    document.getElementById('loss-overfit-test').innerText = overfitTestEval.loss.toFixed(4);

    plotPrediction('pred-overfit-train', noisySplit.train.x, noisySplit.train.y, overfitTrainEval.predictions, 'Trainingsdaten – Overfit');
    plotPrediction('pred-overfit-test', noisySplit.test.x, noisySplit.test.y, overfitTestEval.predictions, 'Testdaten – Overfit');
  }

  runOverfitModel();
