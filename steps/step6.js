// Get Data
async function getData() {
  const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');  
  const carsData = await carsDataReq.json();  
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null));
  
  return cleaned;
}

// Create Model
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  model.add(tf.layers.dense({ units: 1, useBias: true }))

  return model;
};

// Convert Tensor
function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = data.map(d => d.horsepower); // x
    const labels = data.map(d => d.mpg); // y

    const inputTensor = tf.tensor(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor(labels, [labels.length, 1]);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

// Train Model
async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  })
}

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));
    
    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);
    
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    
    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });
  
  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));
  
  
  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}


async function run() {
  // Step 0: Get Data
  const data = await getData();

  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));
  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values}, 
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  // Step 1: Create Model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);

  // Step 2: Convert data to tensor
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  // Step 3: Train Model
  await trainModel(model, inputs, labels);

  // Step 4: Test Model
  testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);