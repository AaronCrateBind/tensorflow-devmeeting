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
  // Step 3: Train Model
  // Step 4: Test Model
}

document.addEventListener('DOMContentLoaded', run);