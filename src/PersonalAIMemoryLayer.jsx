import React, { useState, useEffect, useRef } from 'react';
import { Send, Brain, Search, Plus, Clock, Target, FileText, Lightbulb, BookOpen, Save } from 'lucide-react';

const PersonalAIMemoryLayer = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [memories, setMemories] = useState([]);
  const [goals, setGoals] = useState([]);
  const [projects, setProjects] = useState([]);
  const [insights, setInsights] = useState([]);
  const [activeTab, setActiveTab] = useState('chat');
  const [journalEntry, setJournalEntry] = useState('');
  const [showMemoryAnimation, setShowMemoryAnimation] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const triggerMemoryAnimation = () => {
    setShowMemoryAnimation(true);
    setTimeout(() => setShowMemoryAnimation(false), 3000);
  };

  const handleJournalSave = async () => {
    if (!journalEntry.trim()) return;

    triggerMemoryAnimation();

    try {
      const prompt = `Extract meaningful memories, insights, goals, and projects from this journal entry:

"${journalEntry}"

Respond with a JSON object in this exact format:
{
  "memories": ["array of specific memories or facts to store"],
  "goals": ["array of goals mentioned or implied"],
  "projects": ["array of projects mentioned or implied"],
  "insights": ["array of insights about the user's thoughts, patterns, or realizations"]
}

CRITICAL: Your entire response must be a single, valid JSON object.`;

      const res = await fetch("http://127.0.0.1:8000/cognitive/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: prompt,
          history: []
        })
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error('HTTP Error Details:', {
          status: res.status,
          statusText: res.statusText,
          body: errorText
        });
        throw new Error(`HTTP error! status: ${res.status} - ${errorText}`);
      }

      const data = await res.json();
      console.log('Chat Backend response:', data); // Debug log

      // Handle different possible response formats
      let parsedResponse;
      if (data.result) {
        parsedResponse = JSON.parse(data.result);
      } else if (data.response !== undefined) {
        // If backend returns response directly (your current format)
        parsedResponse = data;
      } else {
        // Log the actual structure and throw error
        console.error('Unexpected chat response structure:', data);
        throw new Error('Invalid response format from backend');
      }

      // Store the raw journal entry as a memory
      const journalMemory = {
        id: Date.now(),
        content: `Journal Entry: ${journalEntry}`,
        timestamp: new Date(),
        source: 'journal',
        type: 'journal'
      };

      setMemories(prev => [journalMemory, ...prev]);

      // Add extracted data
      if (parsedResponse.memories && parsedResponse.memories.length > 0) {
        const extractedMemories = parsedResponse.memories.map((memory, index) => ({
          id: Date.now() + index + 1,
          content: memory,
          timestamp: new Date(),
          source: 'journal_extracted'
        }));
        setMemories(prev => [...prev, ...extractedMemories]);
      }

      if (parsedResponse.goals && parsedResponse.goals.length > 0) {
        const extractedGoals = parsedResponse.goals.map((goal, index) => ({
          id: Date.now() + index + 1,
          content: goal,
          timestamp: new Date(),
          status: 'active',
          source: 'journal'
        }));
        setGoals(prev => [...prev, ...extractedGoals]);
      }

      if (parsedResponse.projects && parsedResponse.projects.length > 0) {
        const extractedProjects = parsedResponse.projects.map((project, index) => ({
          id: Date.now() + index + 1,
          content: project,
          timestamp: new Date(),
          status: 'active',
          source: 'journal'
        }));
        setProjects(prev => [...prev, ...extractedProjects]);
      }

      if (parsedResponse.insights && parsedResponse.insights.length > 0) {
        const extractedInsights = parsedResponse.insights.map((insight, index) => ({
          id: Date.now() + index + 1,
          content: insight,
          timestamp: new Date(),
          source: 'journal'
        }));
        setInsights(prev => [...prev, ...extractedInsights]);
      }

      setJournalEntry('');
    } catch (error) {
      console.error('Error processing journal entry:', error);
      // Add error handling UI feedback here if needed
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize with welcome message
  useEffect(() => {
    setMessages([{
      id: 1,
      type: 'ai',
      content: "Hello! I'm your personal AI memory layer. I'll remember everything we discuss and help you think through your goals, projects, and decisions. What's on your mind today?",
      timestamp: new Date()
    }]);
  }, []);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      // Prepare conversation history (include the new user message)
      const conversationHistory = [...messages, userMessage].slice(-10).map(msg => msg.content);

      // Prepare a simpler prompt for the chat
      const prompt = `You are a personal AI memory layer and cognitive partner. 

Current user message: "${currentInput}"

Based on this message and our conversation history, respond as their intelligent cognitive partner.

Respond with a JSON object in this exact format:
{
  "response": "Your conversational response to the user",
  "newMemories": ["array of new memories to store"],
  "newGoals": ["array of new goals identified"],
  "newProjects": ["array of new projects identified"],
  "newInsights": ["array of insights or connections you've made"],
  "suggestedActions": ["array of suggested next actions"]
}

CRITICAL: Your entire response must be a single, valid JSON object.`;

      const requestPayload = {
        prompt: prompt,
        history: conversationHistory
      };

      console.log('Chat request payload:', {
        prompt: prompt.substring(0, 100) + '...',
        history: conversationHistory
      }); // Debug log with consistent data

      const res = await fetch("http://127.0.0.1:8000/cognitive/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestPayload)
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error('HTTP Error Details:', {
          status: res.status,
          statusText: res.statusText,
          body: errorText
        });
        throw new Error(`HTTP error! status: ${res.status} - ${errorText}`);
      }

      const data = await res.json();
      console.log('Chat backend response:', data); // Debug log

      // Handle different possible response formats
      let parsedResponse;

      // If the response is already in the expected format
      if (data.response !== undefined) {
        parsedResponse = data;
      }
      // If the response is wrapped in a result field
      else if (data.result) {
        try {
          parsedResponse = JSON.parse(data.result);
        } catch (parseError) {
          console.error('Failed to parse result field:', parseError);
          throw new Error('Invalid JSON in result field');
        }
      }
      // If backend returns different format, try to adapt
      else if (data.memories !== undefined || data.goals !== undefined) {
        parsedResponse = {
          response: data.response || "I received your message.",
          newMemories: data.memories || [],
          newGoals: data.goals || [],
          newProjects: data.projects || [],
          newInsights: data.insights || [],
          suggestedActions: data.suggestedActions || []
        };
      }
      else {
        console.error('Unexpected response structure:', data);
        throw new Error('Invalid response format from backend');
      }

      // Add AI message
      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: parsedResponse.response || "I received your message but couldn't generate a proper response.",
        timestamp: new Date(),
        suggestedActions: parsedResponse.suggestedActions || []
      };

      setMessages(prev => [...prev, aiMessage]);

      // Update memories, goals, projects, and insights
      if (parsedResponse.newMemories && parsedResponse.newMemories.length > 0) {
        triggerMemoryAnimation();
        setMemories(prev => [...prev, ...parsedResponse.newMemories.map((memory, index) => ({
          id: Date.now() + index + 2,
          content: memory,
          timestamp: new Date(),
          source: 'conversation'
        }))]);
      }

      if (parsedResponse.newGoals && parsedResponse.newGoals.length > 0) {
        setGoals(prev => [...prev, ...parsedResponse.newGoals.map((goal, index) => ({
          id: Date.now() + index + 2,
          content: goal,
          timestamp: new Date(),
          status: 'active',
          source: 'conversation'
        }))]);
      }

      if (parsedResponse.newProjects && parsedResponse.newProjects.length > 0) {
        setProjects(prev => [...prev, ...parsedResponse.newProjects.map((project, index) => ({
          id: Date.now() + index + 2,
          content: project,
          timestamp: new Date(),
          status: 'active',
          source: 'conversation'
        }))]);
      }

      if (parsedResponse.newInsights && parsedResponse.newInsights.length > 0) {
        setInsights(prev => [...prev, ...parsedResponse.newInsights.map((insight, index) => ({
          id: Date.now() + index + 2,
          content: insight,
          timestamp: new Date(),
          source: 'conversation'
        }))]);
      }

    } catch (error) {
      console.error('Error processing message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: `I apologize, but I encountered an error: ${error.message}. Please try again.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const TabButton = ({ id, icon: Icon, label, count = 0 }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${activeTab === id
        ? 'bg-blue-600 text-white'
        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
        }`}
    >
      <Icon size={18} />
      <span className="font-medium">{label}</span>
      {count > 0 && (
        <span className="bg-white bg-opacity-20 text-xs px-2 py-1 rounded-full">
          {count}
        </span>
      )}
    </button>
  );

  return (
    <div className="h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex">
      {/* Sidebar */}
      <div className="w-80 bg-white shadow-lg p-6 overflow-y-auto">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
            <Brain className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">Cerebrum</h1>
            <p className="text-sm text-gray-600">Your AI cognitive partner</p>
          </div>
        </div>

        <div className="space-y-4">
          <TabButton id="chat" icon={Send} label="Chat" />
          <TabButton id="journal" icon={BookOpen} label="Journal" />
          <TabButton id="memories" icon={FileText} label="Memories" count={memories.length} />
          <TabButton id="goals" icon={Target} label="Goals" count={goals.length} />
          <TabButton id="projects" icon={Plus} label="Projects" count={projects.length} />
          <TabButton id="insights" icon={Lightbulb} label="Insights" count={insights.length} />
        </div>

        {/* Context Panel */}
        <div className="mt-8 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-900 mb-3">Quick Context</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Memories:</span>
              <span className="font-medium">{memories.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Active Goals:</span>
              <span className="font-medium">{goals.filter(g => g.status === 'active').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Projects:</span>
              <span className="font-medium">{projects.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Insights:</span>
              <span className="font-medium">{insights.length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative">
        {/* Neural Glow Animation Background */}
        {showMemoryAnimation && (
          <div className="absolute inset-0 pointer-events-none z-0">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-400/10 via-purple-500/15 to-pink-400/10 animate-pulse"></div>
            <div className="absolute inset-0 opacity-30">
              <div className="absolute top-1/4 left-1/4 w-32 h-32 bg-blue-400/20 rounded-full animate-ping"></div>
              <div className="absolute top-2/3 right-1/3 w-24 h-24 bg-purple-400/20 rounded-full animate-ping delay-500"></div>
              <div className="absolute bottom-1/4 left-1/2 w-20 h-20 bg-pink-400/20 rounded-full animate-ping delay-1000"></div>
            </div>
            <div className="absolute inset-0 bg-gradient-to-br from-transparent via-white/5 to-transparent animate-pulse"></div>
          </div>
        )}

        <div className="relative z-10 flex-1 flex flex-col">
          {activeTab === 'chat' && (
            <>
              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-3xl p-4 rounded-lg ${message.type === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-white text-gray-900 shadow-sm'
                        }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      {message.suggestedActions && message.suggestedActions.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <p className="text-sm text-gray-600 mb-2">Suggested actions:</p>
                          <ul className="space-y-1">
                            {message.suggestedActions.map((action, idx) => (
                              <li key={idx} className="text-sm text-blue-600">• {action}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div className="text-xs opacity-70 mt-2">
                        {formatTime(message.timestamp)}
                      </div>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white p-4 rounded-lg shadow-sm">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-6 bg-white border-t">
                <div className="flex gap-4">
                  <textarea
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Tell me about your thoughts, goals, or what you're working on..."
                    className="flex-1 p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows="3"
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={isLoading || !inputValue.trim()}
                    className="px-6 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send size={20} />
                  </button>
                </div>
              </div>
            </>
          )}

          {activeTab === 'journal' && (
            <div className="flex-1 overflow-y-auto p-6">
              <div className="max-w-4xl mx-auto">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Personal Journal</h2>
                  <p className="text-gray-600">Write your thoughts, reflections, and experiences. Your AI will automatically extract and remember key insights.</p>
                </div>

                <div className="bg-white rounded-lg shadow-sm p-6">
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Today's Entry
                    </label>
                    <textarea
                      value={journalEntry}
                      onChange={(e) => setJournalEntry(e.target.value)}
                      placeholder="What's on your mind today? Share your thoughts, experiences, goals, challenges, or reflections..."
                      className="w-full h-64 p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div className="flex justify-between items-center">
                    <div className="text-sm text-gray-500">
                      {journalEntry.length} characters
                    </div>
                    <button
                      onClick={handleJournalSave}
                      disabled={!journalEntry.trim()}
                      className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      <Save size={18} />
                      Save & Process
                    </button>
                  </div>
                </div>

                <div className="mt-8">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Journal Entries</h3>
                  <div className="space-y-4">
                    {memories.filter(m => m.type === 'journal').length === 0 ? (
                      <p className="text-gray-600">No journal entries yet. Start writing above!</p>
                    ) : (
                      memories.filter(m => m.type === 'journal').map((entry) => (
                        <div key={entry.id} className="bg-white p-4 rounded-lg shadow-sm border-l-4 border-blue-400">
                          <p className="text-gray-900">{entry.content.replace('Journal Entry: ', '')}</p>
                          <div className="text-xs text-gray-500 mt-2">
                            {new Date(entry.timestamp).toLocaleDateString()} at {formatTime(new Date(entry.timestamp))}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'memories' && (
            <div className="flex-1 overflow-y-auto p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Your Memories</h2>
              <div className="space-y-4">
                {memories.length === 0 ? (
                  <p className="text-gray-600">No memories stored yet. Start chatting to build your memory layer!</p>
                ) : (
                  memories.map((memory) => (
                    <div key={memory.id} className="bg-white p-4 rounded-lg shadow-sm">
                      <p className="text-gray-900">{memory.content}</p>
                      <div className="flex justify-between items-center mt-3 text-xs text-gray-500">
                        <span>Source: {memory.source}</span>
                        <span>{formatTime(new Date(memory.timestamp))}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {activeTab === 'goals' && (
            <div className="flex-1 overflow-y-auto p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Your Goals</h2>
              <div className="space-y-4">
                {goals.length === 0 ? (
                  <p className="text-gray-600">No goals identified yet. Share your aspirations in the chat!</p>
                ) : (
                  goals.map((goal) => (
                    <div key={goal.id} className="bg-white p-4 rounded-lg shadow-sm">
                      <div className="flex items-center justify-between">
                        <p className="text-gray-900 font-medium">{goal.content}</p>
                        <span className={`px-2 py-1 rounded-full text-xs ${goal.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                          {goal.status}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mt-2">
                        Added: {formatTime(new Date(goal.timestamp))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {activeTab === 'projects' && (
            <div className="flex-1 overflow-y-auto p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Your Projects</h2>
              <div className="space-y-4">
                {projects.length === 0 ? (
                  <p className="text-gray-600">No projects identified yet. Mention your projects in the chat!</p>
                ) : (
                  projects.map((project) => (
                    <div key={project.id} className="bg-white p-4 rounded-lg shadow-sm">
                      <div className="flex items-center justify-between">
                        <p className="text-gray-900 font-medium">{project.content}</p>
                        <span className={`px-2 py-1 rounded-full text-xs ${project.status === 'active' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                          {project.status}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mt-2">
                        Added: {formatTime(new Date(project.timestamp))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {activeTab === 'insights' && (
            <div className="flex-1 overflow-y-auto p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">AI Insights</h2>
              <div className="space-y-4">
                {insights.length === 0 ? (
                  <p className="text-gray-600">No insights generated yet. Keep chatting to build deeper understanding!</p>
                ) : (
                  insights.map((insight) => (
                    <div key={insight.id} className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
                      <div className="flex items-start gap-3">
                        <Lightbulb className="text-yellow-600 mt-1" size={18} />
                        <div className="flex-1">
                          <p className="text-gray-900">{insight.content}</p>
                          <div className="text-xs text-gray-500 mt-2">
                            Generated: {formatTime(new Date(insight.timestamp))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PersonalAIMemoryLayer;