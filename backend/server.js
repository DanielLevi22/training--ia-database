import express from 'express';
import cors from 'cors'
import { getCustomerInfo } from './database_api.js';
import { getAIResponse } from './ai_api.js';

const app = express();
app.use(cors());
app.use(express.json());

const port = 3000;

const chatHistory = {};

app.post("/support", async (req, res) => {
  const email = req.body.email;
  const message = req.body.message;

  //const email = "Feliciano.Moraes@live.com";

  if (!(email in chatHistory)) {
    chatHistory[email] = []
  }

  chatHistory[email].push({
    role: "user",
    parts: [{ text: message }]
  });

  const customerInfo = await getCustomerInfo(email);
  const response = await getAIResponse(customerInfo, chatHistory[email]);

  chatHistory[email].push({
    role: "model",
    parts: [{ text: response }]
  });

  res.send(JSON.stringify({
    response: response
  }));
});

app.listen(port, () => {
  console.log(`servidor ouvindo na porta ${port}`);
})