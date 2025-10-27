import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader, BookOpen, AlertTriangle } from 'lucide-react';
import MessageBubble from './MessageBubble';
import SourceCitation from './SourceCitation';
import { chatAPI } from '../services/api';

const ChatInterface = ({ hasDocuments }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    
    setMessages(prev => [...prev, { 
      type: 'user', 
      content: userMessage 
    }]);

    setLoading(true);

    try {
      const response = await chatAPI.sendMessage(userMessage, conversationId);
      
      if (!conversationId) {
        setConversationId(response.conversation_id);
      }

      setMessages(prev => [...prev, {
        type: 'assistant',
        content: response.answer,
        sources: response.sources,
        isContextBased: response.is_context_based
      }]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        sources: [],
        isContextBased: false
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const suggestedQuestions = [
    "What are the fundamental rights in the Constitution of Nepal?",
    "Who can become the Prime Minister of Nepal?",
    "What is the structure of the Federal Parliament?",
    "What are the directive principles and policies?",
  ];

  const handleSuggestionClick = (question) => {
    setInput(question);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-full bg-dark-500 rounded-2xl shadow-dark-lg overflow-hidden border border-dark-300 backdrop-blur-lg bg-opacity-90">
      {/* No Documents Warning */}
      {!hasDocuments && (
        <div className="bg-warning/10 border-b border-warning/20 px-4 py-3">
          <div className="flex items-center space-x-2 text-warning">
            <AlertTriangle className="w-5 h-5" />
            <p className="text-sm font-medium">
              No documents indexed. Please upload a PDF to start asking questions.
            </p>
          </div>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 chat-scrollbar">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-2xl px-4">
              <div className="w-24 h-24 bg-linear-to-br from-accent-500 via-purple-600 to-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-dark-lg">
                <BookOpen className="w-12 h-12 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-dark-50 mb-3">
                Welcome to LegalDocRAG
              </h3>
              <p className="text-dark-200 mb-8 leading-relaxed">
                Ask any question about your legal documents and get accurate answers with source citations from the Constitution of Nepal.
              </p>

              {hasDocuments && (
                <div className="space-y-3">
                  <p className="text-sm font-semibold text-dark-100 mb-3">Try asking:</p>
                  <div className="grid gap-2">
                    {suggestedQuestions.map((question, index) => (
                      <button
                        key={index}
                        onClick={() => handleSuggestionClick(question)}
                        className="text-left bg-dark-400/30 hover:bg-dark-400/50 border border-dark-300 hover:border-accent-500/50 rounded-lg px-4 py-3 text-sm text-dark-100 transition-all duration-200 hover:shadow-dark backdrop-blur-sm"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div>
            {messages.map((message, index) => (
              <div key={index}>
                <MessageBubble
                  message={message.content}
                  isUser={message.type === 'user'}
                />
                {message.type === 'assistant' && message.sources && message.sources.length > 0 && (
                  <div className="mb-4 ml-14">
                    <SourceCitation sources={message.sources} />
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex justify-start mb-4">
                <div className="flex items-center space-x-3 bg-dark-400/30 px-5 py-3 rounded-2xl border border-dark-300 shadow-dark backdrop-blur-sm">
                  <Loader className="w-5 h-5 text-accent-400 animate-spin" />
                  <span className="text-dark-100 font-medium">Analyzing documents...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-dark-300 p-4 bg-dark-400/20 backdrop-blur-sm">
        <div className="flex items-end space-x-3 max-w-4xl mx-auto">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={hasDocuments ? "Ask a question about your legal documents..." : "Please upload a document first..."}
              disabled={!hasDocuments || loading}
              rows="1"
              className="w-full resize-none rounded-xl border-2 border-dark-300 focus:border-accent-500 bg-dark-400/30 px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-accent-500/20 disabled:bg-dark-400/10 disabled:cursor-not-allowed transition-all duration-200 text-dark-50 placeholder-dark-200"
              style={{ maxHeight: '120px', minHeight: '48px' }}
            />
          </div>
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading || !hasDocuments}
            className="bg-linear-to-r from-accent-600 to-purple-600 hover:from-accent-700 hover:to-purple-700 disabled:from-dark-300 disabled:to-dark-400 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-all duration-200 shadow-dark hover:shadow-dark-lg transform hover:scale-105 disabled:transform-none"
          >
            <Send className="w-6 h-6" />
          </button>
        </div>
        <p className="text-xs text-dark-200 mt-2 text-center">
          Press Enter to send • Shift + Enter for new line
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;