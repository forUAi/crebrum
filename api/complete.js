// /api/complete.js
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

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

  if (!response.ok) {
    const text = await response.text();
    console.error("❌ OpenAI API Error:", response.status, text);
    return res.status(500).json({ result: "I'm sorry, I couldn't generate a response." });
  }

  const data = await response.json();
  const result = data.choices?.[0]?.message?.content || "Sorry, no reply.";
  res.status(200).json({ result });
}
