const express = require('express');
const path = require('path');
const open = require('open');
const app = express();

const PORT = process.env.PORT || 3000;

const staticPath = process.pkg ? path.join(process.cwd(), 'public') : path.join(__dirname, 'public');

app.use(express.static(staticPath));

app.listen(PORT, async () => {
  const url = `http://localhost:${PORT}`;
  console.log(`App running at ${url}`);
  await open(url);
});