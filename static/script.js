document.getElementById("experiment-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form submission

    const activation = document.getElementById("activation").value;
    const lr = parseFloat(document.getElementById("lr").value);
    const stepNum = parseInt(document.getElementById("step_num").value);

    // Validation checks
    const acts = ["relu", "tanh", "sigmoid"];
    if (!acts.includes(activation)) {
        alert("Please choose from relu, tanh, sigmoid.");
        return;
    }

    if (isNaN(lr)) {
        alert("Please enter a valid number for learning rate.");
        return;
    }

    if (isNaN(stepNum) || stepNum <= 0) {
        alert("Please enter a positive integer for Number of Training Steps.");
        return;
    }

    // Show loading screen
    const loadingDiv = document.getElementById("loading");
    const resultsDiv = document.getElementById("results");
    const resultImg = document.getElementById("result_gif");

    loadingDiv.style.display = "block";
    resultsDiv.style.display = "none";
    resultImg.style.display = "none";

    // Submit the form data to the server
    try {
        const response = await fetch("/run_experiment", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ activation: activation, lr: lr, step_num: stepNum }),
        });

        if (response.ok) {
            const data = await response.json();

            // Set and display the results
            if (data.result_gif) {
                resultImg.src = `/${data.result_gif}`;
                resultImg.onload = function () {
                    // Hide loading and show results when GIF loads
                    loadingDiv.style.display = "none";
                    resultsDiv.style.display = "block";
                    resultImg.style.display = "block";
                };
            } else {
                throw new Error("GIF generation failed.");
            }
        } else {
            throw new Error("Server error occurred.");
        }
    } catch (error) {
        console.error("Error running experiment:", error);
        alert("An error occurred while running the experiment. Please try again.");
        loadingDiv.style.display = "none"; // Hide loading screen on error
    }
});
