// /api/complete.js
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const { prompt } = req.body;

  if (!prompt) {
    return res.status(400).json({ 
      result: JSON.stringify({
        response: "Please provide a prompt.",
        newMemories: [],
        newGoals: [],
        newProjects: [],
        newInsights: [],
        suggestedActions: []
      })
    });
  }

  try {
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
            content: "You are an intelligent memory assistant. You must respond with valid JSON only. Never include any text outside of the JSON structure. The JSON must match the exact format requested by the user."
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
      
      // Return a valid JSON structure even on error
      return res.status(200).json({ 
        result: JSON.stringify({
          response: "I'm sorry, I encountered an error and couldn't process your request. Please try again.",
          newMemories: [],
          newGoals: [],
          newProjects: [],
          newInsights: [],
          suggestedActions: []
        })
      });
    }

    const data = await response.json();
    let result = data.choices?.[0]?.message?.content || "";
    
    // Clean up the response to ensure it's valid JSON
    result = result.trim();
    
    // Remove any markdown code blocks if present
    if (result.startsWith('```json')) {
      result = result.replace(/```json\n?/, '').replace(/\n?```$/, '');
    }
    if (result.startsWith('```')) {
      result = result.replace(/```\n?/, '').replace(/\n?```$/, '');
    }
    
    // Validate that it's actually JSON
    try {
      JSON.parse(result);
    } catch (parseError) {
      console.error("❌ Invalid JSON from OpenAI:", result);
      console.error("Parse error:", parseError);
      
      // Return a fallback response
      result = JSON.stringify({
        response: "I received your message but had trouble formatting my response. Could you please try again?",
        newMemories: [],
        newGoals: [],
        newProjects: [],
        newInsights: [],
        suggestedActions: []
      });
    }
    
    res.status(200).json({ result });
    
  } catch (error) {
    console.error("❌ Server Error:", error);
    
    // Return a valid JSON structure even on server error
    return res.status(200).json({ 
      result: JSON.stringify({
        response: "I'm sorry, I encountered a server error. Please try again later.",
        newMemories: [],
        newGoals: [],
        newProjects: [],
        newInsights: [],
        suggestedActions: []
      })
    });
  }
}