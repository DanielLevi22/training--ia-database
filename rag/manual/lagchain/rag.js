import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { pull } from "langchain/hub";
import { StateGraph, Annotation } from "@langchain/langgraph";
import dotenv from 'dotenv';
dotenv.config();


// 1️⃣ Carrega documentos da web
const url = "https://eloquentjavascript.net/1st_edition/print.html";
const loader = new CheerioWebBaseLoader(url, { selector: ".block" });
const docs = await loader.load();

// 2️⃣ Divide os documentos em chunks
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1500,
    chunkOverlap: 300
});
const allSplits = await splitter.splitDocuments(docs);

// 3️⃣ Cria embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_GEMINI_API_KEY,
    model: "text-embedding-004",
    taskType: "RETRIEVAL_DOCUMENT"
});

// 4️⃣ Inicializa Chroma com persistência
const persistDir = "../vector_database/chroma";
const collectionName = "javascript-book-gemini-embeddings-v2";
const vectorStore = new Chroma(embeddings, {
    collectionName,
    persistDirectory: persistDir
});

// 5️⃣ Popula a collection apenas se estiver vazia
const existingDocs = await vectorStore.similaritySearch("teste", 1).catch(() => []);
if (!existingDocs || existingDocs.length === 0) {
    await vectorStore.addDocuments(allSplits);
    console.log("✅ Collection populada com embeddings");
} else {
    console.log("ℹ️ Collection já possui documentos, pulando addDocuments");
}

// 6️⃣ Carrega prompt e configura LLM
const promptTemplate = await pull("rlm/rag-prompt");
const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.5-flash",
    apiKey: process.env.GOOGLE_GEMINI_API_KEY,
});

// 7️⃣ Funções de retrieve e generate
async function retrieve(state) {
    const retrievedDocs = await vectorStore.similaritySearch(state.question, 3); // 3 mais relevantes
    return { docs: retrievedDocs };
}

async function generate(state) {
    const context = state.docs.map(doc => doc.pageContent).join("\n");
    const prompt = await promptTemplate.invoke({
        question: state.question,
        context,
        instructions: "Responda em português."
    });
    const response = await llm.invoke(prompt);
    return { answer: response };
}

// 8️⃣ Define annotations do grafo
const StateAnnotation = Annotation.Root({
    question: Annotation,
    docs: Annotation,
    answer: Annotation
});

// 9️⃣ Cria grafo
const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

//  🔟 Executa fluxo manualmente
const question = "como funciona uma variável?";
const retrievedDocs = await retrieve({ question });
const response = await generate({ question, docs: retrievedDocs.docs });

const inputs = { question: "como fazer uma recursao ?" }
console.log(await graph.invoke(inputs))
// console.log("📌 Resposta do LLM:\n", response.answer.content);



async function getAnswer(question) {
    const inputs = { question: question };
    return graph.invoke(inputs).then(state => state.answer.content);
}




// Roda o script
export { getAnswer };