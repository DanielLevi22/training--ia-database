import { textToHTML } from "../../utils";
import "./UserMessage.css";

export default function UserMessage({ text }) {
  return (
    <>
      <div className="user-message">{textToHTML(text)}</div>
    </>
  );
}
