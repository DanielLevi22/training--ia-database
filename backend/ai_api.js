import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
dotenv.config();

const genai = new GoogleGenAI({ apiKey: process.env.GOOGLE_GEMINI_API_KEY });

function getDaysSincePurchase(purchase) {
  const today = new Date();
  const diffInMilis = today - purchase.date;

  return Math.floor(diffInMilis / 1000 / 60 / 60 / 24);
}

function getCustomerAge(customer) {
  const today = new Date();
  const diffWithBirthDate = today - customer.birth_date;

  return (new Date(diffWithBirthDate)).getFullYear() - 1970;
}

function getPurchasesString(purchases) {
  let purchasesString = "";

  for (let purchase of purchases) {
    purchasesString += `
        - ${purchase.product}:
            - Preço: ${purchase.price}
            - Status: ${purchase.status}
            - Dias desde a compra: ${getDaysSincePurchase(purchase)}`
  }

  return purchasesString;
}

const systemInstruction = (customer, purchases) => `
Você é um atendente de uma empresa de e-commerce. Você está conversando com clientes que
podem ter dúvidas sobre suas compras recentes no site. Responda os clientes de forma amigável.

Não informe nada a respeito de você para o cliente, diga apenas que você é um atendente virtual.

Caso o cliente pergunte sobre algo não relacionado à empresa ou aos nossos serviços, indique que
não pode ajudá-lo com isso. Caso o cliente pergunte sobre algo relacionado à empresa, mas que não
é explicitamente sobre suas compras passadas, direcione ele ao atendimento humano pelo número: (11)1234-5678.

Altere o tom das suas respostas de acordo com a idade do cliente. Se o cliente for jovem, dialogue de
forma mais informal. Se o cliente for idoso, trate-o com o devido respeito.

Se o cliente reclamar sobre o atraso nas suas compras, verifique se a compra excedeu o SLA de entrega
de acordo com a região do cliente:

NORTE: 10 dias
NORDESTE: 7 dias
SUL: 5 dias
CENTRO-OESTE: 5 dias
SUDESTE: 2 dias

Você não pode realizar nenhuma ação a não ser responder perguntas sobre os dados a seguir. Caso o
cliente necessite de alguma ação por parte da empresa (como contestar compras), direcione-o ao suporte.

<CLIENTE>
Nome: ${customer.first_name} ${customer.last_name}
email: ${customer.email}
idade: ${getCustomerAge(customer)}
estado: ${customer.state}

<COMPRAS>
${getPurchasesString(purchases)}
`;

async function getAIResponse(customerInfo, message) {
  const instruction = systemInstruction(customerInfo.customer, customerInfo.purchases);

  const response = await genai.models.generateContent({
    model: 'gemini-2.0-flash',
    config: {
      systemInstruction: instruction,
    },
    contents: message
  });

  return response.candidates[0].content.parts[0].text;
}

export { getAIResponse }