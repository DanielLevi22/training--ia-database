import { useRef } from "react";
import "./LoginBox.css";

export default function Loginbox({ setEmail }) {
    const emailInput = useRef(null);

    function login() {
        const email = emailInput.current.value;

        if (email.length === 0) {
            return;
        }

        setEmail(email);
    }
    return <>
        <div class="login-background">
            <div className="login-div">
                <input className="login-input" ref={emailInput}></input>
                <button className="login-button" onClick={login}>Login</button>
            </div>
        </div>

    </>
}