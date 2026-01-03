
// Simple placeholder frontend: serves a static index and a small WS endpoint demo
const express = require('express');
const app = express();
const port = 3000;
app.get('/', (req, res) => res.send('<h3>Eagle Vision Frontend Placeholder</h3><p>Connect to streaming WS for frames.</p>'));
app.listen(port, ()=> console.log('Frontend placeholder listening on', port));
