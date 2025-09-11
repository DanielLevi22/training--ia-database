import AIMessage from "../AIMessage/AIMessage";
import UserMessage from "../UserMessage/UserMessage";
import "./Chat.css";

export default function Chat({ messages, isThinking, setCanSendNewMessage }) {
  const messagesWithPlaceholder =
    messages.length % 2 == 0
      ? messages
      : [...messages, { type: "ai", text: "" }];

  function allowNewMessages() {
    if (!isThinking) {
      setCanSendNewMessage(true);
    }
  }

  function messageToHTML(message, idx) {
    if (message.type === "user") {
      return <UserMessage key={idx} text={message.text} />;
    } else {
      const isLastMessage = idx == messagesWithPlaceholder.length - 1;
      return (
        <AIMessage
          key={idx}
          text={message.text}
          isThinking={isThinking}
          finishTypingCallback={allowNewMessages}
          isLastMessage={isLastMessage}
        />
      );
    }
  }

  return (
    <>
      <div className={`chat-window ${messages.length > 2 ? "reverse" : ""}`}>
        <div className="chat-content">
          {messagesWithPlaceholder.map(messageToHTML)}
        </div>
      </div>
    </>
  );
}
