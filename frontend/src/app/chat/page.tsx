"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Image, Send } from 'lucide-react';

// Mock data and types
interface Message {
  id: string;
  sender: 'user' | 'ai';
  content: string;
  imageUrl?: string | null;
  isEnhanced?: boolean; // Track if this message has been enhanced
}

const MOCK_MESSAGES: Message[] = [
  {
    id: 'm1',
    sender: 'user',
    content: 'Can you show me an example of the 3D laser scanner model that had quality issues?',
    imageUrl: null,
  },
  {
    id: 'm2',
    sender: 'ai',
    content:
      'According to the context information provided in the text file, an example of a scanned 3D head model with data quality issues is Figure 1. This figure shows that missing data appear as non-existing holes in the scanned object and may be caused by occlusion during the measuring process.',
    imageUrl: 'https://tyswhhteurchuzkngqja.supabase.co/storage/v1/object/public/comfit_images/3DLaserScanner_P18_img1.png',
    isEnhanced: false,
  },
  {
    id: 'm3',
    sender: 'user',
    content: 'Thank you.',
    imageUrl: null,
  },
  {
    id: 'm4',
    sender: 'ai',
    content: 'You are welcome! Let me know if you have other questions about anthropometry or product fit.',
    imageUrl: null,
    isEnhanced: false,
  },
];

// Simplified UI components
const Button = ({ children, onClick, disabled, className }: { children: React.ReactNode; onClick: () => void; disabled?: boolean; className?: string }) => (
  <button onClick={onClick} disabled={disabled} className={`px-4 py-2 rounded ${className} ${disabled ? 'bg-gray-500' : 'bg-teal-500 hover:bg-teal-600'}`}>
    {children}
  </button>
);

const Input = ({ value, onChange, placeholder, className, onKeyDown }: { value: string; onChange: (e: React.ChangeEvent<HTMLInputElement>) => void; placeholder: string; className?: string; onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void }) => (
  <input
    value={value}
    onChange={onChange}
    placeholder={placeholder}
    className={`p-2 rounded border ${className || 'bg-gray-700 text-white border-gray-600'}`}
    onKeyDown={onKeyDown}
  />
);

// ChatBubble component
const ChatBubble = ({ message }: { message: Message }) => {
  const isUser = message.sender === 'user';
  const bubbleClass = isUser
    ? 'bg-blue-600 text-white self-end rounded-br-none'
    : 'bg-gray-700 text-white self-start rounded-tl-none';

  return (
    <div className={`flex flex-col max-w-lg mx-auto ${isUser ? 'items-end' : 'items-start'} mb-4`}>
      <div className={`p-4 rounded-xl shadow-lg transition duration-200 ${bubbleClass}`}>
        {message.imageUrl && (
          <div className="mb-4 pt-2">
            <h3 className="text-sm font-semibold mb-2 flex items-center text-gray-300">
              <Image className="w-4 h-4 mr-1 text-teal-400" />
              AI Visual Result
            </h3>
            <img
              src={message.imageUrl}
              alt="AI Generated Content"
              className="rounded-lg max-h-96 w-full object-contain border border-gray-600 shadow-md"
              onError={(e) => {
                e.currentTarget.onerror = null;
                e.currentTarget.src = 'https://placehold.co/400x300/ff0000/ffffff?text=Image+Load+Error';
              }}
            />
          </div>
        )}
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        <div className="text-xs mt-2 opacity-60 text-right">
          {isUser ? 'You' : 'Comfit Copilot'}
        </div>
        {!isUser && !message.isEnhanced && (
          <div className="flex justify-end mt-2">
            <Button onClick={() => handleEnhanceClarity(message.id)} className="bg-green-500 hover:bg-green-600 text-white text-xs px-3 py-1">
              Enhance Clarity
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default function ChatbotUI() {
  const [messages, setMessages] = useState<Message[]>(MOCK_MESSAGES);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Handle sending a message (mocked)
  const handleSendMessage = () => {
    if (!inputValue.trim() || isLoading) return;
    setIsLoading(true);
    const newMessage: Message = { id: `m${Date.now()}`, sender: 'user', content: inputValue, imageUrl: null };
    setMessages((prev) => [...prev, newMessage]);
    setInputValue('');
    setTimeout(() => {
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== newMessage.id),
        newMessage,
        {
          id: `m${Date.now() + 1}`,
          sender: 'ai',
          content: 'This is a mock AI response.',
          imageUrl: null,
          isEnhanced: false,
        },
      ]);
      setIsLoading(false);
    }, 1000);
  };

  // Handle enhancing clarity
  const handleEnhanceClarity = async (messageId: string) => {
    const message = messages.find((m) => m.id === messageId);
    if (!message || message.sender !== 'ai' || message.isEnhanced) return;

    setIsLoading(true);
    try {
      const response = await fetch('/api/enhance-clarity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_content: message.content, message_id: messageId }),
      });

      if (!response.ok) throw new Error('Failed to enhance clarity');
      const data = await response.json();
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId ? { ...m, content: data.enhanced_text, isEnhanced: true } : m
        )
      );
    } catch (error) {
      console.error('Error enhancing clarity:', error);
      alert('Failed to enhance clarity. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-scroll to bottom
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="min-h-screen bg-gray-900 font-sans text-gray-100 p-4 sm:p-8">
      <style>{`
        .chat-container::-webkit-scrollbar {
          width: 8px;
        }
        .chat-container::-webkit-scrollbar-thumb {
          background-color: #4b5563;
          border-radius: 4px;
        }
        .chat-container::-webkit-scrollbar-track {
          background: #1f2937;
        }
      `}</style>

      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-extrabold mb-8 text-teal-400 border-b border-gray-700 pb-2">
          ComFit Chat (Image Feature Live)
        </h1>

        <div className="chat-container h-[70vh] overflow-y-auto p-4 space-y-4 rounded-lg bg-gray-800 shadow-xl" ref={chatContainerRef}>
          {messages.map((msg) => (
            <ChatBubble key={msg.id} message={msg} />
          ))}
          {messages.length === 0 && (
            <div className="text-center text-gray-500 pt-10">Start a new conversation to see the results!</div>
          )}
        </div>

        <div className="mt-6 p-4 bg-gray-800 rounded-lg shadow-xl border border-gray-700">
          <p className="text-sm text-gray-400">
            Backend Status: <span className="text-green-400 font-medium ml-2">200 OK. Image URL generation verified.</span>
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Try a query like: "show me a laser scan" or "generate image" to see the simulated result.
          </p>
        </div>

        <div className="mt-6 flex items-center space-x-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 bg-gray-700 text-white border-gray-600"
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
          />
          <Button onClick={handleSendMessage} disabled={isLoading} className="bg-teal-500 hover:bg-teal-600">
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}