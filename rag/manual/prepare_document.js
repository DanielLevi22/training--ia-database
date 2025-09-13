import * as cheerio from 'cheerio';
import { ChromaClient } from 'chromadb';
const url = "https://eloquentjavascript.net/1st_edition/print.html";

const client = new ChromaClient()

const collection = await client.getOrCreateCollection({
  name: "javascript-book"
})





const htmlString = await fetch(url).then(r => r.text());
const $ = cheerio.load(htmlString);
const documents = $(".block").toArray().map(block => $(block).text());
await collection.add({
  documents: documents,
  ids: documents.map((_, index) => index.toString())
})
