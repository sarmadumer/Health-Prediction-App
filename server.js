const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");

const app = express();
app.use(cors());
app.use(express.json());

app.post("/predict", (req, res) => {
  const input = req.body.values;
  const type = req.body.type || "diabetes";

  const script = type === "heart" ? "heart_predict.py" : "predict.py";
  const python = spawn("python", [script, ...input]);

  let result = "";

  python.stdout.on("data", (data) => result += data.toString());
  python.stderr.on("data", (data) => console.error("Python error:", data.toString()));

  python.on("close", (code) => {
    if (code !== 0) return res.status(500).json({ error: "Python script failed" });
    const finalResult = result.trim();
    res.json({ result: finalResult === "1" ? 
      (type === "diabetes" ? "Diabetic" : "Heart Disease Detected") : 
      (type === "diabetes" ? "You are not diabetic" : "No heart disease detected") });
  });
});

app.listen(3000, () => console.log("Server running at http://localhost:3000"));
