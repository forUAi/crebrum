export default async function handler(req, res) {
  try {
    const { prompt } = req.body;

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are an intelligent memory assistant. Respond in structured JSON only.",
          },
          {
            role: "user",
            content: prompt,
          }
        ],
        temperature: 0.7
      }),
    });

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    // 👇 Fallback to valid JSON if content is missing or invalid
    const result = content || JSON.stringify({
      response: "I'm sorry, I couldn't generate a response.",
      newMemories: [],
      newGoals: [],
      newProjects: [],
      newInsights: [],
      suggestedActions: []
    });

    res.status(200).json({ result });
  } catch (error) {
    console.error("Error in /api/complete:", error);
    res.status(500).json({
      result: JSON.stringify({
        response: "An internal error occurred.",
        newMemories: [],
        newGoals: [],
        newProjects: [],
        newInsights: [],
        suggestedActions: []
      })
    });
  }
}
