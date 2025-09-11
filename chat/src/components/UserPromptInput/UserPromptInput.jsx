import { useState, useRef } from "react";
import { FaArrowUp } from "react-icons/fa";
import "./UserPromptInput.css";

export default function UserPromptInput({
  onSubmit,
  canSubmit,
  conversationStarted,
}) {
  const [buttonVisible, setButtonVisible] = useState(false);
  const userPromptTextArea = useRef(null);

  function handleInput() {
    maybeToggleButton();
    resizeTextArea();
  }

  function maybeToggleButton() {
    setButtonVisible(userPromptTextArea.current.value.length > 0);
  }

  function resizeTextArea() {
    const textArea = userPromptTextArea.current;

    textArea.style.height = "auto";
    textArea.style.height = textArea.scrollHeight + "px";
  }

  function maybeSubmitOnEnter(e) {
    if (e.keyCode === 13 && !e.shiftKey) {
      e.preventDefault();
      maybeSubmitPrompt();
    }
  }

  function maybeSubmitPrompt() {
    const textArea = userPromptTextArea.current;

    if (textArea.value.length == 0) return;
    if (!canSubmit) return;

    onSubmit(textArea.value);
    textArea.value = "";
    resizeTextArea();
    maybeToggleButton();
  }

  return (
    <>
      <div
        className={`user-prompt-holder ${conversationStarted ? "fixed-to-bottom" : ""}`}
      >
        <h1
          className={`welcome-message ${conversationStarted ? "hidden" : ""}`}
        >
          What can I help with?
        </h1>
        <div className="user-prompt-box">
          <textarea
            className="user-prompt-input"
            ref={userPromptTextArea}
            placeholder="Ask anything"
            onInput={handleInput}
            onKeyDown={maybeSubmitOnEnter}
          ></textarea>
          <div className="submit-button-holder">
            <div
              className={`submit-button ${buttonVisible ? "visible" : ""}`}
              onClick={maybeSubmitPrompt}
            >
              <FaArrowUp color="black" />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
