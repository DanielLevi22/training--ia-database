export default async function getResponse(email, prompt) {
  return fetch("http://localhost:3000/support", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      email: email,
      message: prompt
    })
  })
  .then(r => r.json()) // <- tem que ser .json()
  .then(o => o.response);
}