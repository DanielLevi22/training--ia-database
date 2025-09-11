function textToHTML(text) {
  return text.split("\n").map((t, idx) => <p key={idx}>{t}</p>);
}

export { textToHTML };
