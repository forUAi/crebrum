// /api/complete.js - TEST VERSION (replace temporarily)
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const { prompt } = req.body;

  console.log("🔑 API Key exists:", !!process.env.OPENAI_API_KEY);
  console.log("📝 Received prompt:", prompt.substring(0, 100) + "...");

  // Return a test response to verify the structure works
  const testResponse = {
    response: "This is a test response. I can see you're working on something interesting! Let me help you think through it.",
    newMemories: ["Test memory: User is testing the AI system"],
    newGoals: ["Test goal: Get the AI system working properly"],
    newProjects: ["Test project: Personal AI Memory Layer"],
    newInsights: ["Test insight: The user is building a sophisticated AI assistant"],
    suggestedActions: ["Check the logs to debug the OpenAI API issue", "Verify the API key is set correctly"]
  };

  res.status(200).json({ result: JSON.stringify(testResponse) });
}