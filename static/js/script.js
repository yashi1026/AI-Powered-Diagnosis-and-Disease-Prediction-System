// =======================
// 💬 CHATBOT FUNCTIONALITY
// =======================

// Toggle chatbox
function toggleChat() {
    let chat = document.getElementById("chatbox");
    chat.style.display = chat.style.display === "flex" ? "none" : "flex";
}

// Send message
function sendMessage() {
    let input = document.getElementById("userInput");
    let message = input.value.trim();
    let messages = document.getElementById("messages");

    if (message === "") return;

    // Show user message
    messages.innerHTML += `<div><b>You:</b> ${message}</div>`;

    // Bot response
    let reply = getBotResponse(message);
    messages.innerHTML += `<div><b>Bot:</b> ${reply}</div>`;

    input.value = "";

    // Auto scroll
    messages.scrollTop = messages.scrollHeight;
}

// Enter key support
document.addEventListener("DOMContentLoaded", function () {
    let input = document.getElementById("userInput");

    if (input) {
        input.addEventListener("keypress", function (e) {
            if (e.key === "Enter") {
                sendMessage();
            }
        });
    }
});

// Simple chatbot logic
function getBotResponse(msg) {
    msg = msg.toLowerCase();

    if (msg.includes("diabetes"))
        return "Diabetes is a condition where blood sugar levels are high.";

    if (msg.includes("bmi"))
        return "BMI tells if your weight is healthy for your height.";

    if (msg.includes("risk"))
        return "Risk depends on glucose, BMI, and age.";

    if (msg.includes("hello") || msg.includes("hi"))
        return "Hello! How can I assist you today?";

    return "I can help with basic health-related questions.";
}



// =======================
//  SCROLL REVEAL ANIMATION
// =======================

window.addEventListener("scroll", function () {
    let reveals = document.querySelectorAll(".reveal");

    reveals.forEach((el) => {
        let top = el.getBoundingClientRect().top;
        let windowHeight = window.innerHeight;

        if (top < windowHeight - 100) {
            el.classList.add("active");
        }
    });
});

// Run once on page load
window.addEventListener("load", function () {
    let reveals = document.querySelectorAll(".reveal");

    reveals.forEach((el) => {
        let top = el.getBoundingClientRect().top;
        let windowHeight = window.innerHeight;

        if (top < windowHeight - 100) {
            el.classList.add("active");
        }
    });
});

