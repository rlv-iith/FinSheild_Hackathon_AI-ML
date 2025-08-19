const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');

const app = express();
app.use(cors()); // Enable cross-origin requests
app.use(express.json()); // Enable parsing of JSON request bodies

// URL for our Python ML service, using its Docker service name
const ML_SERVICE_URL = 'http://prediction-service:5000/predict';

// Define the main API endpoint
app.post('/api/check-risk', async (req, res) => {
  console.log('Received request at /api/check-risk');
  try {
    const userData = req.body;
    
    console.log('Forwarding data to ML service:', userData);
    
    // Make an internal POST request to the Python prediction service
    const mlResponse = await axios.post(ML_SERVICE_URL, userData);
    
    console.log('Received response from ML service.');
    
    // Send the prediction back to the frontend
    res.json(mlResponse.data);

  } catch (error) {
    console.error("Error communicating with ML service:", error.message);
    res.status(500).json({ error: 'Could not get a prediction from the ML service.' });
  }
});

const PORT = 8080;
app.listen(PORT, () => {
  console.log(`API Gateway server running on port ${PORT}`);
});