import express from 'express';
import cors from 'cors';
import { getAnswer } from './rag.js';

const app = express();
app.use(cors());
app.use(express.json());

const port = 3000;

app.post("/question_answering", async (req, res) => {
    console.log(req.body.question);
    const answer = await getAnswer(req.body.question);

    res.send({ response: answer });
});

app.listen(port, () => {
    console.log(`servidor ouvindo na porta ${port}`);
})