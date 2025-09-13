import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
dotenv.config();

import { ChromaClient } from "chromadb";

const client = new ChromaClient();
const collection = await client.getOrCreateCollection({
  name: "javascript-book"
});


const genai = new GoogleGenAI({ apiKey: process.env.GOOGLE_GEMINI_API_KEY });

const userQuestion = "what is a function?";

async function retrieve(question) {
  return collection.query({
    queryTexts: [question],
    nresults: 5
  });
}


const retrievedChunks = (await retrieve(userQuestion)).documents[0].join("\n\n=======\n\n");

const systemInstruction = `
Você é um expert em Javascript que vai responder a uma pergunta do usuário.

Responda a pergunta com base nos seguintes trechos retirados do livro "Eloquent Javascript".
Referencie em sua resposta os trechos abaixo, deixe explicíto onde começa a referência ao livro.
Adicione esses trechos à resposta caso necessário.

${retrievedChunks}
`;

const response = await genai.models.generateContent({
  model: "gemini-2.0-flash",
  config: {
    systemInstruction: systemInstruction
  },
  contents: userQuestion
});

console.log(response.candidates[0].content.parts[0].text);