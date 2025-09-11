import { useState, useEffect } from "react";
import { textToHTML } from "../../utils";
import "./AIMessage.css";
import ThinkingIndicator from "../ThinkingIndicator/ThinkingIndicator";

export default function AIMessage({
  text = "",
  isThinking,
  finishTypingCallback,
  isLastMessage,
}) {
  const [displayText, setDisplayText] = useState("");
  const [displayTextIdx, setDisplayTextIdx] = useState(0);
  const [shouldFill, setShouldFill] = useState(false);

  useEffect(() => {
    if (displayTextIdx < text.length) {
      const write = setTimeout(() => {
        setDisplayText((prevText) => prevText + text[displayTextIdx]);
        setDisplayTextIdx((idx) => idx + 1);
      }, 5);

      return () => clearTimeout(write);
    } else {
      finishTypingCallback();
    }
  }, [text, displayTextIdx]);

  useEffect(() => {
    setShouldFill(isLastMessage);
  }, [isLastMessage]);

  return (
    <>
      <div className={`ai-chat-message ${shouldFill ? "fill" : ""}`}>
        {isThinking && isLastMessage ? <ThinkingIndicator /> : null}
        {textToHTML(displayText)}
      </div>
    </>
  );
}
