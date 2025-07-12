// /api/complete.js
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const { prompt } = req.body;

  if (!prompt) {
    return res.status(200).json({ 
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

  // Check if API key exists
  if (!process.env.OPENAI_API_KEY) {
    console.error("❌ OPENAI_API_KEY not found in environment variables");
    return res.status(200).json({ 
      result: JSON.stringify({
        response: "OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable.",
        newMemories: [],
        newGoals: [],
        newProjects: [],
        newInsights: [],
        suggestedActions: ["Add OPENAI_API_KEY to Vercel environment variables"]
      })
    });
  }

  try {
    console.log("🔑 API Key exists:", !!process.env.OPENAI_API_KEY);
    console.log("📝 Prompt length:", prompt.length);
    
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
        temperature: 0.7,
        max_tokens: 1000
      }),
    });

    console.log("🌐 OpenAI Response Status:", response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error("❌ OpenAI API Error:", response.status, errorText);
      
      let errorMessage = "OpenAI API error occurred.";
      
      // Parse specific error messages
      try {
        const errorData = JSON.parse(errorText);
        if (errorData.error) {
          errorMessage = errorData.error.message || errorMessage;
        }
      } catch (e) {
        errorMessage = errorText.substring(0, 100);
      }
      
      return res.status(200).json({ 
        result: JSON.stringify({
          response: `API Error (${response.status}): ${errorMessage}`,
          newMemories: [],
          newGoals: [],
          newProjects: [],
          newInsights: [],
          suggestedActions: [
            "Check your OpenAI API key",
            "Verify you have available credits",
            "Try again in a few minutes"
          ]
        })
      });
    }

    const data = await response.json();
    let result = data.choices?.[0]?.message?.content || "";
    
    console.log("📋 Raw OpenAI Response:", result);
    
    if (!result) {
      console.error("❌ Empty response from OpenAI");
      return res.status(200).json({ 
        result: JSON.stringify({
          response: "Received empty response from OpenAI. Please try again.",
          newMemories: [],
          newGoals: [],
          newProjects: [],
          newInsights: [],
          suggestedActions: []
        })
      });
    }
    
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
      const parsed = JSON.parse(result);
      console.log("✅ Valid JSON response received");
      
      // Ensure all required fields exist
      const validatedResponse = {
        response: parsed.response || "I processed your message.",
        newMemories: Array.isArray(parsed.newMemories) ? parsed.newMemories : [],
        newGoals: Array.isArray(parsed.newGoals) ? parsed.newGoals : [],
        newProjects: Array.isArray(parsed.newProjects) ? parsed.newProjects : [],
        newInsights: Array.isArray(parsed.newInsights) ? parsed.newInsights : [],
        suggestedActions: Array.isArray(parsed.suggestedActions) ? parsed.suggestedActions : []
      };
      
      result = JSON.stringify(validatedResponse);
      
    } catch (parseError) {
      console.error("❌ Invalid JSON from OpenAI:", result);
      console.error("Parse error:", parseError);
      
      // Return a fallback response
      result = JSON.stringify({
        response: "I received your message but had trouble formatting my response. Let me try to help anyway based on what you said.",
        newMemories: [prompt.substring(0, 200)], // Store the original message as a memory
        newGoals: [],
        newProjects: [],
        newInsights: [],
        suggestedActions: ["Try rephrasing your message", "The AI is having formatting issues"]
      });
    }
    
    res.status(200).json({ result });
    
  } catch (error) {
    console.error("❌ Server Error:", error);
    
    // Return a valid JSON structure even on server error
    return res.status(200).json({ 
      result: JSON.stringify({
        response: `Server error: ${error.message}`,
        newMemories: [],
        newGoals: [],
        newProjects: [],
        newInsights: [],
        suggestedActions: ["Try again", "Check server logs"]
      })
    });
  }
}