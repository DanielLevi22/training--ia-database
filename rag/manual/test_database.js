
import { ChromaClient } from "chromadb";

const client = new ChromaClient();
const collection = await client.getOrCreateCollection({
  name: "javascript-book"
});

console.log(await collection.query({
  queryTexts: ["how functions work"],
  nresults: 2
}))