const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
const PORT = 8080;
const ML_SERVICE_URL = 'http://ml_service:5000/predict';

app.use(express.json());
app.use(cors());

// A simple welcome route
app.get('/', (req, res) => {
  res.send('Welcome to the Alt-Credit Scoring API Gateway!');
});

// The main scoring endpoint
app.post('/score', async (req, res) => {
  console.log('Received scoring request with data:', req.body);

  // Validate incoming data
  if (!req.body || typeof req.body.monthly_income_rs === 'undefined') {
    return res.status(400).json({ error: 'Invalid input data. Missing key features.' });
  }

  try {
    // Forward the request to the Python ML service
    const response = await axios.post(ML_SERVICE_URL, req.body);
    
    // Return the prediction from the ML service to the client
    res.json(response.data);
  } catch (error) {
    console.error('Error calling ML service:', error.message);
    res.status(500).json({ error: 'Could not connect to the ML service.' });
  }
});

app.listen(PORT, () => {
  console.log(`API Gateway server is running on http://localhost:${PORT}`);
});