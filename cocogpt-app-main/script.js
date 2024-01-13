const inputBox = document.getElementById("user_input_key");
inputBox.addEventListener("keyup", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        const submitButton = document.querySelector("button[aria-label='Send']");
        submitButton.click();
    }
});

