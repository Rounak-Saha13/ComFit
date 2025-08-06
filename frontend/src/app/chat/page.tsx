"use client";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
// imports
import { SetStateAction, useEffect, useRef, useState } from "react";
import {
  Search,
  Settings,
  ArrowUp,
  Plus,
  X,
  Menu,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  Copy,
  ChevronDown,
  Edit,
  MoreVertical,
  ArrowRight,
  ArrowLeft,
  Check,
  User as UserIcon,
  Mic,
  ChevronRight,
  Paperclip,
  Brain,
  VectorSquare,
  Globe,
  MessageSquare,
  Database,
  HelpCircle,
} from "lucide-react";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import ManageProfile from "@/components/ui/mange-profile";
import FormattedContent from "@/components/ui/FormattedContent";

// for authentication
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabaseClient";
import type { User } from "@supabase/supabase-js";
import { v4 as uuidv4 } from "uuid";

// types
interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  thinkingTime?: number;
  feedback?: number | null;
  isThinking?: boolean;
}

interface BranchItem {
  messages: Message[];
  branchId: string | null;
}

interface HistoryResponse {
  messages: Message[];
  branchesByEditId: Record<
    string,
    Array<{ messages: Message[]; branchId: string }>
  >;
  currentBranchIndexByEditId: Record<string, number>;
}

interface ConversationState {
  messages: Message[];
  originalMessages?: Message[];
  editAtId?: string;
  branchesByEditId?: Record<string, BranchItem[]>;
  currentBranchIndexByEditId?: Record<string, number>;
}

interface LoadHistoryResult extends HistoryResponse {
  activeBranchId: string | null;
}

export default function ChatbotUI() {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  // states for chat/history
  const [history, setHistory] = useState<ConversationState[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentBranchId, setCurrentBranchId] = useState<string | null>(null);

  // states for ui/input
  const [inputValue, setInputValue] = useState("");
  const [showLeftSidebar, setShowLeftSidebar] = useState(true);
  const [showRightSidebar, setShowRightSidebar] = useState(true);
  const [openAccordions, setOpenAccordions] = useState<string[]>([]);
  const [isAwaitingResponse, setIsAwaitingResponse] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [mechanisticInterpretability, setMechanisticInterpretability] =
    useState(false);

  // states for enhanced Plus dropdown and active features
  const [showPlusDropdown, setShowPlusDropdown] = useState(false);
  const [activeFeatures, setActiveFeatures] = useState<string[]>([]);
  const [openSubmenu, setOpenSubmenu] = useState<string | null>(null);

  // states for editing
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState<string>("");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Engine type toggle state
  const [engineType, setEngineType] = useState<"chat" | "query">("chat");

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const currentMessages = history[currentIndex]?.messages || [];
  const isWelcomeState = currentMessages.length === 0;
  const editTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  // rate limiting
  const [showLimitModal, setShowLimitModal] = useState(false);

  // authentication
  const router = useRouter();
  const supabase = createClient();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // uploading functionality
  const [uploading, setUploading] = useState(false);

  // collapsible thinking
  const [showThoughts, setShowThoughts] = useState<Record<string, boolean>>({});
  const toggleThoughts = (id: string) =>
    setShowThoughts((prev) => ({ ...prev, [id]: !prev[id] }));

  // copy functionality
  const [justCopiedId, setJustCopiedId] = useState<string | null>(null);

  // cancel functionality
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);

  // regenerate functionality
  const [isRegenerating, setIsRegenerating] = useState(false);

  // states for model selection
  const [selectedModel, setSelectedModel] = useState("gemma3:latest");
  const [models, setModels] = useState<string[]>([]);

  // states for RAG and retrieval methods
  const [selectedRagMethod, setSelectedRagMethod] = useState(
    "No Specific RAG Method"
  );
  const [selectedRetrievalMethod, setSelectedRetrievalMethod] =
    useState("local context only");

  // right sidebar controls
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful AI assistant for comfort and fitting clothing"
  );
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [specDecoding, setSpecDecoding] = useState(false);
  const [strategy, setStrategy] = useState("no-workflow");
  const [preset, setPreset] = useState("CFIR");

  // manage profile modal
  const [showProfileModal, setShowProfileModal] = useState(false);

  // AI thinking time - removed unused state variable

  const openProfileModal = () => {
    setShowUserMenu(false);
    setShowProfileModal(true);
  };
  const closeProfileModal = () => setShowProfileModal(false);

  // Responsive sidebar behavior
  useEffect(() => {
    const handleResize = () => {
      const isSmallScreen = window.innerWidth < 768; // md breakpoint
      const isMediumScreen = window.innerWidth < 1024; // lg breakpoint

      if (isSmallScreen) {
        // On small screens, close both sidebars
        setShowLeftSidebar(false);
        setShowRightSidebar(false);
      } else if (isMediumScreen) {
        // On medium screens, close right sidebar but keep left sidebar
        setShowRightSidebar(false);
      }
    };

    // Set initial state based on current screen size
    handleResize();

    // Add event listener
    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // hook for session
  useEffect(() => {
    // fetch current session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // listen for future auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [router, supabase]);

  // hook for initializing history
  useEffect(() => {
    const initialHistory = conversations.map(() => ({
      messages: [],
      editAtId: undefined,
    }));
    setHistory(initialHistory);
    setCurrentIndex(0);
  }, []);

  // hook for auto‐scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentMessages]);

  // Autosize textbox when editing message
  useEffect(() => {
    if (editTextareaRef.current) {
      const textarea = editTextareaRef.current;
      textarea.style.height = "auto"; // reset
      textarea.style.height = textarea.scrollHeight + "px"; // grow to fit
    }
  }, [editingText, editingId]);

  // load conversations
  useEffect(() => {
    if (loading || typeof window === "undefined") return;

    async function loadConversations() {
      try {
        // get supabase session
        const {
          data: { session },
        } = await supabase.auth.getSession();

        if (session?.access_token) {
          headers.Authorization = `Bearer ${session.access_token}`;
        }

        const res = await fetch(`${API_URL}/api/conversations`, {
          headers,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          console.error("Fetch error:", res.status, err);
          return;
        }

        const data = await res.json();
        if (!Array.isArray(data)) {
          console.error("Expected an array but got:", data);
          return;
        }

        if (data.length === 0) {
          const newConvo = await handleNewChat();
          if (newConvo) {
            setConversations([newConvo]);
            setHistory([{ messages: [], editAtId: undefined }]);
            setCurrentIndex(0);
          }
          return;
        }

        const currentConvId = conversations[currentIndex]?.id;
        setConversations(data);
        const newIndex = currentConvId
          ? data.findIndex((c) => c.id === currentConvId)
          : 0;

        setHistory((prev) =>
          data.map((_, idx) =>
            idx === newIndex && prev[currentIndex]
              ? prev[currentIndex]
              : { messages: [], editAtId: undefined }
          )
        );
        setCurrentIndex(Math.max(0, newIndex));

        // Load the current conversation's history if it exists
        if (data.length > 0 && newIndex >= 0) {
          const currentConversation = data[newIndex];
          try {
            const {
              messages,
              branchesByEditId = {},
              currentBranchIndexByEditId = {},
            } = await loadHistory(currentConversation.id);

            // Convert backend field names to frontend field names
            const convertedMessages = messages.map((msg: any) => {
              const converted = {
                ...msg,
                thinkingTime: msg.thinking_time,
              };
              if (msg.sender === "ai") {
                console.log("DEBUG: Loading AI message from backend:", {
                  id: msg.id,
                  thinking_time: msg.thinking_time,
                  thinkingTime: converted.thinkingTime,
                });
              }
              return converted;
            });

            // Convert branch messages as well
            const convertedBranchesByEditId: Record<string, BranchItem[]> = {};
            for (const [editId, branches] of Object.entries(branchesByEditId)) {
              convertedBranchesByEditId[editId] = branches.map(
                (branch: any) => ({
                  ...branch,
                  messages: branch.messages.map((msg: any) => ({
                    ...msg,
                    thinkingTime: msg.thinking_time,
                  })),
                })
              );
            }

            setHistory((prev) => {
              const newHist = [...prev];
              newHist[newIndex] = {
                messages: convertedMessages,
                originalMessages: convertedMessages,
                branchesByEditId: convertedBranchesByEditId,
                currentBranchIndexByEditId,
              };
              return newHist;
            });
          } catch (err) {
            console.error("Error loading current conversation history:", err);
          }
        }
      } catch (err) {
        console.error("Error loading conversations:", err);
      } finally {
        setLoading(false);
      }
    }

    loadConversations();
  }, [user, loading]);

  // fetch models list
  useEffect(() => {
    async function loadModels() {
      console.log("DEBUG: Loading models from:", `${API_URL}/api/models`);
      try {
        const res = await fetch(`${API_URL}/api/models`);
        console.log("DEBUG: Models response status:", res.status);

        if (!res.ok) {
          const errorText = await res.text();
          console.error("DEBUG: Models error response:", errorText);
          throw new Error(errorText);
        }

        const data = await res.json();
        console.log("DEBUG: Models response data:", data);
        const { models } = data;
        console.log("DEBUG: Available models:", models);
        setModels(models);
        if (models.length > 0) {
          console.log("DEBUG: Setting initial model to:", models[0]);
          setSelectedModel(models[0]);
        }
      } catch (err) {
        console.error("DEBUG: Error loading models:", err);
      }
    }
    loadModels();
  }, []);

  const handleFeedback = async (messageId: string, value: number) => {
    // check value
    const current = history[currentIndex].messages.find(
      (m) => m.id === messageId
    )?.feedback;
    const newValue = current === value ? null : value;

    // update UI
    setHistory((prev) => {
      const copy = [...prev];
      copy[currentIndex] = {
        ...copy[currentIndex],
        messages: copy[currentIndex].messages.map((m) =>
          m.id === messageId ? { ...m, feedback: newValue } : m
        ),
      };
      return copy;
    });

    try {
      // get supabase session
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }

      const res = await fetch(`${API_URL}/api/messages/${messageId}/feedback`, {
        method: "POST",
        headers,
        body: JSON.stringify({ rating: newValue }),
      });
    } catch (error) {
      console.error("Could not save feedback:", error);
    }
  };

  const handleRegenerate = async (aiMessageId: string) => {
    // Set regenerating state for this specific message
    setIsRegenerating(true);

    setHistory((prev) =>
      prev.map((convo, idx) =>
        idx === currentIndex
          ? {
              ...convo,
              messages: convo.messages.map((m) =>
                m.id === aiMessageId
                  ? { ...m, content: "", isThinking: true }
                  : m
              ),
            }
          : convo
      )
    );

    // Create abort controller for regeneration
    const controller = new AbortController();
    setAbortController(controller);
    setIsAwaitingResponse(true);

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }

      const msgs = history[currentIndex].messages;
      const idx = msgs.findIndex((m) => m.id === aiMessageId);
      const slice = msgs.slice(0, idx);
      const guestHistory = slice.map((m) => ({
        role: m.sender,
        content: m.content,
      }));

      const res = await fetch(
        `${API_URL}/api/messages/${aiMessageId}/regenerate`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({
            conversation_id: conversations[currentIndex].id,
            history: guestHistory,
            model: selectedModel,
            preset,
            temperature,
            rag_method: selectedRagMethod,
            retrieval_method: selectedRetrievalMethod,
          }),
          signal: controller.signal,
        }
      );

      if (!res.ok) throw new Error("Regeneration failed");

      // backend returns the updated Message object
      const updatedMsg = await res.json();

      setHistory((prev) => {
        const copy = [...prev];
        // replace the old AI message with the regenerated one
        copy[currentIndex].messages = copy[currentIndex].messages.map((m) =>
          m.id === aiMessageId
            ? {
                id: updatedMsg.id,
                content: updatedMsg.content,
                sender: "ai",
                thinkingTime: updatedMsg.thinking_time,
                feedback: updatedMsg.feedback,
                isThinking: false,
              }
            : m
        );
        return copy;
      });
    } catch (err: any) {
      if (err.name === "AbortError") {
        // Handle cancellation
        setHistory((prev) => {
          const copy = [...prev];
          copy[currentIndex].messages = copy[currentIndex].messages.map((m) =>
            m.id === aiMessageId
              ? { ...m, content: "Regeneration cancelled", isThinking: false }
              : m
          );
          return copy;
        });
        return;
      }
      console.error("Regeneration error:", err);

      // Restore original content on error
      setHistory((prev) => {
        const copy = [...prev];
        copy[currentIndex].messages = copy[currentIndex].messages.map((m) =>
          m.id === aiMessageId
            ? {
                ...m,
                content: "Regeneration failed. Please try again.",
                isThinking: false,
              }
            : m
        );
        return copy;
      });
    } finally {
      setIsRegenerating(false);
      setAbortController(null);
      setIsAwaitingResponse(false);
    }
  };

  const loadHistory = async (convId: string): Promise<LoadHistoryResult> => {
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }

    const res = await fetch(
      `${API_URL}/api/messages/conversation/${convId}/history`,
      {
        method: "GET",
        headers,
      }
    );

    if (!res.ok) throw new Error("Failed to load history");

    const historyData = (await res.json()) as HistoryResponse;
    // get active branchID
    let activeBranchId: string | null = null;
    for (const [editId, branchList] of Object.entries(
      historyData.branchesByEditId
    )) {
      const idx = historyData.currentBranchIndexByEditId[editId];
      if (branchList[idx]?.branchId) {
        activeBranchId = branchList[idx].branchId;
        break;
      }
    }

    return {
      ...historyData,
      activeBranchId,
    };
  };

  // load messages immediately when convo is clicked
  const handleConversationClick = async (idx: number, convId: string) => {
    const {
      messages,
      branchesByEditId = {},
      currentBranchIndexByEditId = {},
    } = await loadHistory(convId);

    // Convert backend field names to frontend field names
    const convertedMessages = messages.map((msg: any) => ({
      ...msg,
      thinkingTime: msg.thinking_time,
    }));

    // Convert branch messages as well
    const convertedBranchesByEditId: Record<string, BranchItem[]> = {};
    for (const [editId, branches] of Object.entries(branchesByEditId)) {
      convertedBranchesByEditId[editId] = branches.map((branch: any) => ({
        ...branch,
        messages: branch.messages.map((msg: any) => ({
          ...msg,
          thinkingTime: msg.thinking_time,
        })),
      }));
    }

    setHistory((prev) => {
      const newHist = [...prev];
      newHist[idx] = {
        messages: convertedMessages,
        originalMessages: convertedMessages,
        branchesByEditId: convertedBranchesByEditId,
        currentBranchIndexByEditId,
      };
      return newHist;
    });

    setCurrentIndex(idx);
    setCurrentBranchId(null);
  };

  // move conversation history
  const moveConversationToTop = (idx: number) => {
    setConversations((prev) => {
      const arr = [...prev];
      const [chosenConv] = arr.splice(idx, 1);
      return [chosenConv, ...arr];
    });

    setHistory((prev) => {
      const arr = [...prev];
      const [chosenHist] = arr.splice(idx, 1);
      return [chosenHist, ...arr];
    });

    setCurrentIndex(0);
  };

  // generate AI response using the selected model
  const generateAIResponse = async (messages: any[], model: string) => {
    console.log("DEBUG: generateAIResponse called");
    console.log("DEBUG: API_URL:", API_URL);

    const {
      data: { session },
    } = await supabase.auth.getSession();

    const payload = {
      conversation_id: conversations[currentIndex].id,
      messages: messages.map((m) => ({
        id: m.id,
        conversation_id: conversations[currentIndex].id,
        sender: m.sender,
        content: m.content,
        thinking_time: m.thinkingTime,
        feedback: m.feedback,
        model: selectedModel,
        preset: preset,
        system_prompt: systemPrompt,
        speculative_decoding: specDecoding,
        temperature: temperature,
        top_p: topP,
        strategy: strategy,
        rag_method: selectedRagMethod,
        retrieval_method: selectedRetrievalMethod,
      })),
      system_prompt: systemPrompt,
      model: selectedModel,
      temperature,
      top_p: topP,
      speculative_decoding: specDecoding,
      strategy,
      preset,
      rag_method: selectedRagMethod,
      retrieval_method: selectedRetrievalMethod,
    };

    console.log("DEBUG: Request payload:", payload);
    console.log("DEBUG: Request URL:", `${API_URL}/api/chat`);
    console.log("DEBUG: Selected model in generateAIResponse:", selectedModel);
    console.log("DEBUG: Model parameter passed to generateAIResponse:", model);

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
      console.log("DEBUG: Auth token present");
    } else {
      console.log("DEBUG: No auth token");
    }

    console.log("DEBUG: Request headers:", headers);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      console.log("DEBUG: Response status:", response.status);
      console.log(
        "DEBUG: Response headers:",
        Object.fromEntries(response.headers.entries())
      );

      if (!response.ok) {
        const errorText = await response.text();
        console.error("DEBUG: Response error text:", errorText);
        console.error("DEBUG: Failed payload:", payload);
        throw new Error(
          `⚠️ Failed to generate AI response. Status: ${response.status}, Error: ${errorText}`
        );
      }

      const responseData = await response.json();
      console.log("DEBUG: Response data:", responseData);
      return responseData;
    } catch (error) {
      console.error("DEBUG: Fetch error:", error);
      throw error;
    }
  };

  var limitReached = false;
  const hasMultipleBranches = (messageId: string): boolean => {
    const conv = history[currentIndex];
    return (conv.branchesByEditId?.[messageId]?.length ?? 0) > 1;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isAwaitingResponse) return;

    const {
      data: { session },
    } = await supabase.auth.getSession();
    let conversationId: string;

    if (conversations.length === 0) {
      const newConversation = await handleNewChat();
      if (!newConversation) return;
      conversationId = newConversation.id;
    } else {
      conversationId = conversations[currentIndex].id;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: "user",
    };

    setHistory((prev) => {
      const copy = [...prev];
      copy[currentIndex] = {
        ...copy[currentIndex],
        messages: [...copy[currentIndex].messages, userMessage],
        editAtId: undefined,
      };
      return copy;
    });
    setInputValue("");
    const controller = new AbortController();
    setAbortController(controller);
    setIsAwaitingResponse(true);

    // if its a branch
    if (currentBranchId) {
      const messagesForAI = [
        ...(history[currentIndex]?.messages || []),
        userMessage,
      ];
      const { result: aiResp, duration } = await generateAIResponse(
        messagesForAI,
        selectedModel
      );

      const newAi: Message = {
        id: Date.now().toString(),
        content: aiResp,
        sender: "ai",
        thinkingTime: duration,
      };

      const updatedBranchMessages = [...messagesForAI, newAi];
      const backendMessages = updatedBranchMessages.map(toBackendMessage);

      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }
      await fetch(`${API_URL}/api/messages/branches/${currentBranchId}`, {
        method: "PATCH",
        headers,
        body: JSON.stringify(backendMessages),
      });

      setHistory((prev) => {
        const copy = [...prev];
        const conv = copy[currentIndex];
        let found = false;
        let editId = null;
        let branchIdx = null;
        if (conv.branchesByEditId) {
          for (const [eid, branches] of Object.entries(conv.branchesByEditId)) {
            const idx = branches.findIndex(
              (b) => b.branchId === currentBranchId
            );
            if (idx !== -1) {
              editId = eid;
              branchIdx = idx;
              found = true;
              break;
            }
          }
        }
        if (found && editId !== null && branchIdx !== null) {
          const updatedBranches = [...(conv.branchesByEditId?.[editId] || [])];
          updatedBranches[branchIdx] = {
            ...updatedBranches[branchIdx],
            messages: updatedBranchMessages,
          };
          copy[currentIndex] = {
            ...conv,
            messages: updatedBranchMessages,
            branchesByEditId: {
              ...conv.branchesByEditId,
              [editId]: updatedBranches,
            },
          };
        } else {
          copy[currentIndex] = {
            ...conv,
            messages: updatedBranchMessages,
          };
        }
        return copy;
      });

      setIsAwaitingResponse(false);
      return;
    }
    // normal flow - when not in a branch
    const payload = {
      conversation_id: conversationId,
      messages: [
        ...(history[currentIndex]?.messages || []).map((m) => ({
          id: m.id,
          conversation_id: conversationId,
          sender: m.sender,
          content: m.content,
          thinking_time: m.thinkingTime,
          feedback: m.feedback,
          model: selectedModel,
          preset,
          system_prompt: systemPrompt,
          speculative_decoding: specDecoding,
          temperature,
          top_p: topP,
          strategy,
          rag_method: selectedRagMethod,
          retrieval_method: selectedRetrievalMethod,
        })),
        {
          id: userMessage.id,
          conversation_id: conversationId,
          sender: "user",
          content: userMessage.content,
        },
      ],
      system_prompt: systemPrompt,
      model: selectedModel,
      temperature,
      top_p: topP,
      speculative_decoding: specDecoding,
      strategy,
      preset,
      rag_method: selectedRagMethod,
      retrieval_method: selectedRetrievalMethod,
    };

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    try {
      console.log("DEBUG: Sending chat request to:", `${API_URL}/api/chat`);
      console.log("DEBUG: Chat payload:", payload);
      console.log("DEBUG: Selected model in payload:", payload.model);
      console.log("DEBUG: Selected model state:", selectedModel);

      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      console.log("DEBUG: Chat response status:", res.status);
      console.log(
        "DEBUG: Chat response headers:",
        Object.fromEntries(res.headers.entries())
      );

      // rate limit pop up
      if (res.status === 429) {
        console.log("DEBUG: Rate limit reached");
        limitReached = true;
        setShowLimitModal(true);
        return;
      }

      if (!res.ok) {
        const errorText = await res.text();
        console.error("DEBUG: Chat API error response:", errorText);
        throw new Error(`Chat API error: ${res.status} - ${errorText}`);
      }

      const responseData = await res.json();
      console.log("DEBUG: Chat response data:", responseData);
      const { result: aiText, duration, ai_message } = responseData;
      console.log("DEBUG: AI message from response:", ai_message);
      console.log(
        "DEBUG: AI message thinking_time:",
        ai_message?.thinking_time
      );
      console.log("DEBUG: Duration from response:", duration);
      const newMessage = {
        id: ai_message.id,
        content: ai_message.content,
        sender: "ai" as const,
        thinkingTime: ai_message.thinking_time || duration, // Fallback to duration if thinking_time is not available
      };
      console.log("DEBUG: New message being added:", newMessage);

      setHistory((prev) => {
        const copy = [...prev];
        copy[currentIndex] = {
          ...copy[currentIndex],
          messages: [...copy[currentIndex].messages, newMessage],
          editAtId: undefined,
        };
        console.log(
          "DEBUG: Updated history with new message:",
          copy[currentIndex].messages
        );
        console.log(
          "DEBUG: Last message thinking time:",
          copy[currentIndex].messages[copy[currentIndex].messages.length - 1]
            ?.thinkingTime
        );
        return copy;
      });

      setIsAwaitingResponse(false);

      // title generation for first message
      const wasFirst = history[currentIndex]?.messages.length === 0;
      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }
      if (wasFirst) {
        const titleRes = await fetch(`${API_URL}/api/title`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            conversation_id: conversations[currentIndex].id,
            user_message: userMessage.content,
            ai_response: aiText,
          }),
        });
        const { title: finalTitle } = await titleRes.json();
        // Update local list
        setConversations((prev) =>
          prev.map((c, i) =>
            i === currentIndex ? { ...c, title: finalTitle } : c
          )
        );

        // Persist the new title
        if (session?.access_token) {
          headers.Authorization = `Bearer ${session.access_token}`;
        }
        await fetch(`${API_URL}/api/conversations/${conversationId}`, {
          method: "PATCH",
          headers,
          body: JSON.stringify({ title: finalTitle }),
        });
      }

      moveConversationToTop(currentIndex);
    } catch (err: any) {
      if (err.name === "AbortError") {
        return;
      }
      console.error("Error in handleSendMessage:", err);
    } finally {
      setAbortController(null);
      setIsAwaitingResponse(false);
    }
  };

  // cancel current AI request
  const handleCancel = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      const cancelMsg: Message = {
        id: uuidv4(),
        content: "Message cancelled",
        sender: "ai",
      };
      setHistory((prev) => {
        const copy = [...prev];
        const conv = copy[currentIndex];
        copy[currentIndex] = {
          ...conv,
          messages: [...conv.messages, cancelMsg],
        };
        return copy;
      });
    }
    setIsAwaitingResponse(false);
  };

  // Edit message
  const startEdit = (messageId: string, currentContent: string) => {
    setEditingId(messageId);
    setEditingText(currentContent);
  };

  // Cancel editing
  const cancelEdit = () => {
    setEditingId(null);
    setEditingText("");
  };

  const getBranchIndex = (messageId: string): number => {
    const conv = history[currentIndex];
    return conv.currentBranchIndexByEditId?.[messageId] ?? 0;
  };

  function toBackendMessage(msg: Message) {
    return {
      ...msg,
      thinking_time: msg.thinkingTime,
      feedback: msg.feedback ?? null,
      thinkingTime: undefined,
    };
  }
  // branch logic
  const commitEdit = async (messageId: string) => {
    const trimmed = editingText.trim();
    if (!trimmed) return;

    const originalMessages =
      history[currentIndex].originalMessages || history[currentIndex].messages;
    const msgIdx = originalMessages.findIndex((m) => m.id === messageId);
    if (msgIdx === -1) return;
    const original = originalMessages[msgIdx].content;
    const hasChanged = trimmed !== original;

    setEditingId(null);
    setEditingText("");

    // Update the message content in the original array
    const updatedOriginalMsgs = [...originalMessages];
    updatedOriginalMsgs[msgIdx] = {
      ...updatedOriginalMsgs[msgIdx],
      content: trimmed,
    };

    setHistory((prev) => {
      const copy = [...prev];
      copy[currentIndex] = {
        ...copy[currentIndex],
        messages: updatedOriginalMsgs,
      };
      return copy;
    });
    setIsAwaitingResponse(true);

    const patchMsg = toBackendMessage(updatedOriginalMsgs[msgIdx]);
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    await fetch(`${API_URL}/api/messages/${messageId}`, {
      method: "PATCH",
      headers,
      body: JSON.stringify(patchMsg),
    });

    // Regenerate if message has not changed
    if (!hasChanged) {
      const messagesUpToEdit = updatedOriginalMsgs.slice(0, msgIdx + 1);
      setHistory((prev) => {
        const copy = [...prev];
        copy[currentIndex] = {
          ...copy[currentIndex],
          messages: messagesUpToEdit,
        };
        return copy;
      });

      const { result: aiResp, duration } = await generateAIResponse(
        messagesUpToEdit,
        selectedModel
      );
      const newAi: Message = {
        id: Date.now().toString(),
        content: aiResp,
        sender: "ai",
        thinkingTime: duration,
      };
      const regenerated = [...messagesUpToEdit, newAi];

      setHistory((prev) => {
        const copy = [...prev];
        copy[currentIndex] = {
          ...copy[currentIndex],
          messages: regenerated,
        };
        return copy;
      });

      setIsAwaitingResponse(false);
      return;
    }

    const messagesUpToEdit = updatedOriginalMsgs.slice(0, msgIdx + 1);

    // get AI reply
    const { result: aiResp, duration } = await generateAIResponse(
      messagesUpToEdit,
      selectedModel
    );
    const newAi: Message = {
      id: Date.now().toString(),
      content: aiResp,
      sender: "ai",
      thinkingTime: duration,
    };

    // persist new AI message to Supabase
    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    await fetch(`${API_URL}/api/messages`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        id: newAi.id,
        conversation_id: conversations[currentIndex].id,
        sender: "ai",
        content: newAi.content,
        thinking_time: newAi.thinkingTime,
      }),
    });

    const branchedMessages = [...messagesUpToEdit, newAi];
    const convId = conversations[currentIndex].id;
    const { data: existingAny } = await supabase
      .from("branches")
      .select("id")
      .eq("conversation_id", convId)
      .eq("edit_at_id", messageId);

    if (!existingAny?.length) {
      await supabase.from("branches").insert([
        {
          conversation_id: convId,
          edit_at_id: messageId,
          messages: originalMessages,
        },
      ]);
    }

    const { data: branchData } = await supabase
      .from("branches")
      .insert([
        {
          conversation_id: convId,
          edit_at_id: messageId,
          messages: branchedMessages,
        },
      ])
      .select("id");

    const newBranchId = branchData?.[0]?.id ?? null;

    setHistory((prev) => {
      const newHist = [...prev];
      const conv = newHist[currentIndex];
      const eid = messageId;

      // grab existing branches for this edit, default empty
      const existing = conv.branchesByEditId?.[eid] ?? [];

      const updatedList: BranchItem[] = existing.length
        ? [...existing, { messages: branchedMessages, branchId: newBranchId }]
        : [
            { messages: originalMessages, branchId: null },
            { messages: branchedMessages, branchId: newBranchId },
          ];

      newHist[currentIndex] = {
        ...conv,
        messages: branchedMessages,
        branchesByEditId: {
          ...(conv.branchesByEditId || {}),
          [eid]: updatedList,
        },
        currentBranchIndexByEditId: {
          ...(conv.currentBranchIndexByEditId || {}),
          [eid]: updatedList.length - 1,
        },
        originalMessages: originalMessages,
      };
      return newHist;
    });
    setCurrentBranchId(newBranchId);
    setIsAwaitingResponse(false);
  };

  const activateBranch = async (branchId: string | null) => {
    if (!branchId) return;
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    await fetch(`${API_URL}/api/branches/${branchId}/activate`, {
      method: "POST",
      headers,
    });
  };

  const goToPrev = async (messageId: string) => {
    const conv = history[currentIndex];
    const branches = conv.branchesByEditId?.[messageId] || [];
    const currentIdx = conv.currentBranchIndexByEditId?.[messageId] || 0;

    if (branches.length <= 1) return;
    const prevIdx = (currentIdx - 1 + branches.length) % branches.length;
    const selectedBranch = branches[prevIdx];
    // for debug
    await activateBranch(selectedBranch.branchId);
    console.log("selectedBranch.messages", selectedBranch.messages);
    setHistory((prev) => {
      const newHist = [...prev];
      newHist[currentIndex] = {
        ...conv,
        messages: selectedBranch.messages,
        currentBranchIndexByEditId: {
          ...(conv.currentBranchIndexByEditId || {}),
          [messageId]: prevIdx,
        },
      };
      return newHist;
    });
    setCurrentBranchId(selectedBranch.branchId);
  };

  const goToNext = async (messageId: string) => {
    const conv = history[currentIndex];
    const branches = conv.branchesByEditId?.[messageId] || [];
    const currentIdx = conv.currentBranchIndexByEditId?.[messageId] || 0;

    if (branches.length <= 1) return;
    const nextIdx = (currentIdx + 1) % branches.length;
    const selectedBranch = branches[nextIdx];
    await activateBranch(selectedBranch.branchId);
    setHistory((prev) => {
      const newHist = [...prev];
      newHist[currentIndex] = {
        ...conv,
        messages: selectedBranch.messages,
        currentBranchIndexByEditId: {
          ...(conv.currentBranchIndexByEditId || {}),
          [messageId]: nextIdx,
        },
      };
      return newHist;
    });
    setCurrentBranchId(selectedBranch.branchId);
  };

  // ---------- UI handlers -----------------
  const handleExampleClick = (example: string) => {
    setInputValue(example);
  };

  const toggleAccordion = (value: string) => {
    setOpenAccordions((prev) =>
      prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value]
    );
  };

  // copy
  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setJustCopiedId(id);
      // clear after 1.5s, revert icon
      setTimeout(() => setJustCopiedId(null), 1500);
    } catch (err) {
      console.error("Copy failed", err);
    }
  };

  // upload documents
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    console.log("File selected:", {
      name: file.name,
      size: file.size,
      type: file.type,
      preset: preset,
    });

    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("preset", preset);

    // Debug: Log FormData contents
    console.log("FormData contents:");
    for (const [key, value] of formData.entries()) {
      console.log(key, value);
    }

    const {
      data: { session },
      error,
    } = await supabase.auth.getSession();

    const uploadHeaders: Record<string, string> = {};
    if (session?.access_token) {
      uploadHeaders.Authorization = `Bearer ${session.access_token}`;
      console.log("Auth token present");
    } else {
      console.log("No auth token");
    }

    try {
      console.log("Sending request to:", `${API_URL}/api/documents/append`);
      const res = await fetch(`${API_URL}/api/documents/append`, {
        method: "POST",
        headers: uploadHeaders,
        body: formData,
      });

      console.log("Response status:", res.status);
      console.log(
        "Response headers:",
        Object.fromEntries(res.headers.entries())
      );

      if (res.ok) {
        const responseData = await res.json();
        console.log("Success response:", responseData);
        alert("Document ingested successfully");
      } else {
        const errorText = await res.text();
        console.error("Error response:", errorText);

        // Try to parse as JSON for better error info
        try {
          const errorJson = JSON.parse(errorText);
          console.error("Parsed error:", errorJson);
          alert(
            `Failed to ingest document: ${
              errorJson.detail || errorJson.message || errorText
            }`
          );
        } catch {
          alert(`Failed to ingest document: ${errorText}`);
        }
      }
    } catch (error: any) {
      console.error("Network error:", error);
      alert(`Network error: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  // --------- left sidebar ----------------
  const [conversations, setConversations] = useState<
    { id: string; title: string }[]
  >([]);

  // chat rename
  const startInlineRename = (id: string, currentTitle: string) => {
    setEditingId(id);
    setEditingText(currentTitle);
    setOpenMenuId(null);
  };

  async function commitInlineRename() {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!editingId) return;

    const trimmed = editingText.trim();
    if (!trimmed) return setEditingId(null);

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    const res = await fetch(`${API_URL}/api/conversations/${editingId}`, {
      method: "PATCH",
      headers,
      body: JSON.stringify({ title: trimmed }),
    });
    if (!res.ok) throw new Error("Could not rename conversation");

    setConversations((prev) =>
      prev.map((c) => (c.id === editingId ? { ...c, title: trimmed } : c))
    );
    setEditingId(null);
    setEditingText("");
  }

  // create new chat
  async function handleNewChat() {
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    const res = await fetch(`${API_URL}/api/conversations`, {
      method: "POST",
      headers,
      body: JSON.stringify({ title: "New Chat" }),
    });
    if (!res.ok) throw new Error("Could not create conversation");
    const newConvo = await res.json();

    // push into conversations list
    setConversations((prev) => [newConvo, ...prev]);

    // initialize an empty message history slot for it
    setHistory((prev) => [
      { messages: [], editAtId: undefined, branches: [], currentBranch: 0 },
      ...prev,
    ]);
    setCurrentIndex(0);

    return newConvo;
  }

  // delete chat
  const deleteChat = async (deletingId: string | null) => {
    if (!deletingId) return;
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    const res = await fetch(`${API_URL}/api/conversations/${deletingId}`, {
      method: "DELETE",
      headers,
    });
    if (!res.ok) throw new Error("Failed to delete conversation");

    // remove from both lists by ID
    setConversations((prev) => prev.filter((c) => c.id !== deletingId));
    setHistory((prev) =>
      prev.filter((_, idx) => conversations[idx]?.id !== deletingId)
    );
    setCurrentIndex(0);
    setDeletingId(null);
  };

  // --------- search ------------
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // search chat handlers
  const filteredResults = conversations.filter((conv) =>
    conv.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSearchResultClick = (idx: number) => {
    setCurrentIndex(idx);
    setShowSearchModal(false);
    setSearchQuery("");
  };

  // ---------- user menu / logout ------------
  const handleUserMenuToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowUserMenu((prev) => !prev);
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    router.push("/auth/login");
  };

  // + drop down
  const togglePlusDropdown = () => {
    setShowPlusDropdown(!showPlusDropdown);
    setOpenSubmenu(null);
  };

  const toggleSubmenu = (submenuName: string) => {
    setOpenSubmenu(openSubmenu === submenuName ? null : submenuName);
  };

  const selectFeature = (feature: string) => {
    // Remove any existing features from the same category
    const category = feature.split(" - ")[0];
    const filteredFeatures = activeFeatures.filter(
      (f) => !f.startsWith(category)
    );

    // Add the new feature
    setActiveFeatures([...filteredFeatures, feature]);
    setOpenSubmenu(null);
  };

  const clearFeature = (featureToRemove: string) => {
    setActiveFeatures(activeFeatures.filter((f) => f !== featureToRemove));
  };

  const clearAllFeatures = () => {
    setActiveFeatures([]);
  };

  // loading spinner
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div
      className="flex h-screen bg-background text-foreground overflow-hidden"
      onClick={() => {
        setOpenMenuId(null);
        setShowUserMenu(false);
        setShowPlusDropdown(false);
        setOpenSubmenu(null);
      }}
    >
      {/* Left Sidebar */}
      {showLeftSidebar && (
        <div className="w-56 sm:w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
          <div className="p-3 sm:p-4 border-b border-sidebar-border">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <div className="flex items-center gap-2">
                <div className="w-16 sm:w-20 h-8 sm:h-10 flex items-center justify-center">
                  <Image
                    src="/Innovision-White-Text.svg"
                    alt="Logo"
                    width={100}
                    height={100}
                    className="object-cover"
                  />
                </div>
              </div>
              <button
                onClick={() => setShowLeftSidebar(false)}
                className="text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent p-1 rounded cursor-pointer transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <button
              onClick={handleNewChat}
              className="w-full flex items-center justify-start text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent p-2 rounded cursor-pointer transition-colors text-sm sm:text-base"
            >
              <Plus className="w-4 h-4 mr-2" />
              New chat
            </button>
            <button
              onClick={() => setShowSearchModal(true)}
              className="w-full flex items-center justify-start text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent p-2 rounded mt-2 cursor-pointer transition-colors text-sm sm:text-base"
            >
              <Search className="w-4 h-4 mr-2" />
              Search chats
            </button>
          </div>

          <div className="flex-1 p-3 sm:p-4 overflow-y-auto sidebar-scrollbar">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs sm:text-sm text-sidebar-foreground">
                Recent Conversations
              </span>
            </div>

            {conversations.length === 0 ? (
              <div className="text-xs sm:text-sm text-sidebar-foreground italic">
                No chats yet
              </div>
            ) : (
              conversations.map((conv, idx) => (
                <div
                  key={conv.id}
                  onClick={() => handleConversationClick(idx, conv.id)}
                  className={`flex items-center justify-between p-2 rounded transition-colors
                    ${
                      idx === currentIndex
                        ? "bg-gray-700 text-white"
                        : "text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent"
                    }
                    cursor-pointer
                  `}
                >
                  {/* Title / Inline Input */}
                  {editingId === conv.id ? (
                    <input
                      type="text"
                      value={editingText}
                      autoFocus
                      onChange={(e) => setEditingText(e.target.value)}
                      onBlur={commitInlineRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          commitInlineRename();
                        }
                      }}
                      className="w-full bg-input text-foreground px-2 py-1 rounded border border-border focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  ) : (
                    <span
                      className="text-xs sm:text-sm text-sidebar-foreground cursor-pointer truncate"
                      onClick={() => handleConversationClick(idx, conv.id)}
                    >
                      {conv.title}
                    </span>
                  )}

                  {/* 3-dot menu */}
                  <div className="relative ml-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenMenuId(conv.id);
                      }}
                      className="p-1 text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent rounded cursor-pointer"
                    >
                      <MoreVertical className="w-4 h-4" />
                    </button>

                    {openMenuId === conv.id && (
                      <div className="absolute right-0 mt-1 w-32 bg-gray-900 border border-sidebar-border rounded shadow-lg z-20">
                        <button
                          className="w-full text-left px-4 py-2 text-sm text-sidebar-foreground hover:bg-sidebar-primary flex items-center gap-2 cursor-pointer"
                          onClick={() => startInlineRename(conv.id, conv.title)}
                        >
                          <Edit className="w-4 h-4" />
                          Rename
                        </button>
                        <button
                          className="w-full text-left px-4 py-2 text-sm text-sidebar-foreground hover:bg-sidebar-primary flex items-center gap-2 cursor-pointer"
                          onClick={() => {
                            setOpenMenuId(null);
                            setDeletingId(conv.id);
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="p-3 sm:p-4 border-t border-sidebar-border relative">
            {user ? (
              <div>
                <button
                  onClick={handleUserMenuToggle}
                  className="flex items-center justify-between gap-2 w-full focus:outline-none"
                >
                  <div className="w-12 h-8 bg-black rounded-full flex items-center justify-center">
                    <UserIcon className="w-4 h-4 text-sidebar-foreground" />
                  </div>
                  <span className="block w-full text-left text-xs sm:text-sm text-sidebar-foreground justify-left truncate">
                    {user.user_metadata.full_name}
                  </span>
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-foreground ${
                      showUserMenu ? "rotate-180" : ""
                    }`}
                  />
                </button>

                {showUserMenu && (
                  <div className="absolute bottom-full inset-x-0 ml-2 mb-2 w-9/10 bg-black border border-sidebar-border rounded shadow-lg z-50">
                    <button
                      onClick={openProfileModal}
                      className="w-full text-left px-4 py-2 text-sm text-sidebar-foreground hover:bg-sidebar-primary cursor-pointer"
                    >
                      Manage profile
                    </button>
                    <button
                      onClick={handleLogout}
                      className="w-full text-left px-4 py-2 text-sm text-destructive hover:bg-sidebar-primary cursor-pointer"
                    >
                      Log out
                    </button>
                  </div>
                )}

                <ManageProfile
                  isOpen={showProfileModal}
                  onClose={closeProfileModal}
                  user={user}
                />
              </div>
            ) : (
              <div className="flex space-x-2">
                <Link href="/auth/login">
                  <Button
                    variant="secondary"
                    size="lg"
                    className="cursor-pointer text-blue-400 hover:text-blue-300 border-blue-400 hover:border-blue-300"
                  >
                    Log in
                  </Button>
                </Link>
                <Link href="/auth/signup">
                  <Button
                    size="lg"
                    variant="default"
                    className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    Sign up
                  </Button>
                </Link>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {/* Header */}
        <div className="h-16 border-b border-border flex items-center justify-between px-4 sm:px-6 flex-shrink-0">
          <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
            {!showLeftSidebar && (
              <button
                onClick={() => setShowLeftSidebar(true)}
                className="text-muted-foreground hover:text-foreground hover:bg-accent p-2 rounded"
              >
                <Menu className="w-4 h-4" />
              </button>
            )}

            {/* Product Name and Model Selection */}
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-semibold text-foreground">
                Comfit Copilot
              </h1>

              {/* Model Selection*/}
              <Select
                value={selectedModel}
                onValueChange={(value) => {
                  console.log("DEBUG: Model selection changed to:", value);
                  setSelectedModel(value);
                }}
              >
                <SelectTrigger className="w-32 sm:w-40 bg-transparent border-none shadow-none focus:ring-0 focus:ring-offset-0">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-black">
                  {models.length === 0 ? (
                    <SelectItem value="none" disabled>
                      (no models available)
                    </SelectItem>
                  ) : (
                    models.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            </div>

            {/* RAG Method Selection */}
            <Select
              value={selectedRagMethod}
              onValueChange={setSelectedRagMethod}
            >
              <SelectTrigger className="w-36 sm:w-44 bg-input border-border">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-black">
                <SelectItem value="RAC Enhanced Hybrid RAG">
                  RAC Enhanced Hybrid RAG
                </SelectItem>
                <SelectItem value="Planning Workflow">
                  Planning Workflow
                </SelectItem>
                <SelectItem value="Multi-Step Query Engine">
                  Multi-Step Query Engine
                </SelectItem>
                <SelectItem value="Multi-Strategy Workflow">
                  Multi-Strategy Workflow
                </SelectItem>
                <SelectItem value="No Specific RAG Method">
                  No Specific RAG Method
                </SelectItem>
              </SelectContent>
            </Select>

            {/* Retrieval Method Selection */}
            <Select
              value={selectedRetrievalMethod}
              onValueChange={setSelectedRetrievalMethod}
            >
              <SelectTrigger className="w-32 sm:w-40 bg-input border-border">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-black">
                <SelectItem value="local context only">
                  Local Context Only
                </SelectItem>
                <SelectItem value="Web searched context only">
                  Web Searched Context Only
                </SelectItem>
                <SelectItem value="Hybrid context">Hybrid Context</SelectItem>
                <SelectItem value="Smart retrieval">Smart Retrieval</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {!showRightSidebar && (
            <button
              onClick={() => setShowRightSidebar(true)}
              className="text-muted-foreground hover:text-foreground hover:bg-accent p-2 rounded flex items-center gap-2"
            >
              <Settings className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-h-0">
          {isWelcomeState ? (
            <div className="flex-1 flex flex-col items-center justify-center p-4 sm:p-8">
              <h1 className="text-xl sm:text-2xl lg:text-3xl font-light mb-6 sm:mb-8 lg:mb-12 text-center px-4">
                What would you like to know?
              </h1>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 mb-6 sm:mb-8 max-w-6xl mx-auto px-4 w-full">
                <button
                  className="border border-border text-muted-foreground hover:bg-accent px-3 sm:px-4 py-2 rounded cursor-pointer text-left text-sm sm:text-base"
                  onClick={() =>
                    handleExampleClick(
                      "What is your name and who developed you?"
                    )
                  }
                >
                  What is your name and who developed you?
                </button>
                <button
                  className="border border-border text-muted-foreground hover:bg-accent px-3 sm:px-4 py-2 rounded cursor-pointer text-left text-sm sm:text-base"
                  onClick={() =>
                    handleExampleClick(
                      "Why is bending your knees before lifting safer than keeping your legs straight?"
                    )
                  }
                >
                  Why is bending your knees before lifting safer than keeping
                  your legs straight?
                </button>
                <button
                  className="border border-border text-muted-foreground hover:bg-accent px-3 sm:px-4 py-2 rounded cursor-pointer text-left text-sm sm:text-base"
                  onClick={() =>
                    handleExampleClick(
                      "Explain why slipping on a wet surface leads to a fall—what forces and frictional changes are at play?"
                    )
                  }
                >
                  Explain why slipping on a wet surface leads to a fall—what
                  forces and frictional changes are at play?
                </button>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-4 sm:p-6 chat-scrollbar">
              <div className="max-w-4xl mx-auto space-y-4 sm:space-y-6">
                {currentMessages.map((message, idx) => {
                  // Debug: Log message details for AI messages
                  if (message.sender === "ai") {
                    console.log("DEBUG: Rendering AI message:", {
                      id: message.id,
                      content: message.content.substring(0, 50) + "...",
                      thinkingTime: message.thinkingTime,
                      hasThinkingTime: message.thinkingTime != null,
                    });
                  }

                  return (
                    <div key={message.id} className="flex gap-3 sm:gap-4">
                      {message.sender === "user" ? (
                        <div className="ml-auto max-w-[280px] sm:max-w-xs lg:max-w-md flex flex-col items-end">
                          {/* User bubble */}
                          <div className="bg-gray-600 text-white p-3 rounded-2xl rounded-br-md whitespace-pre-wrap break-words max-w-full shadow-sm">
                            {editingId === message.id ? (
                              <div className="space-y-2">
                                <textarea
                                  ref={editTextareaRef}
                                  value={editingText}
                                  onChange={(e) =>
                                    setEditingText(e.target.value)
                                  }
                                  className="bg-input text-foreground p-2 rounded border border-border focus:outline-none focus:ring-2 focus:ring-primary resize-none overflow-hidden"
                                  style={{ width: "400px" }}
                                  rows={1}
                                  autoFocus
                                />
                                <div className="flex gap-2">
                                  <button
                                    onClick={() => commitEdit(message.id)}
                                    className="px-3 py-1 bg-primary hover:bg-primary/90 text-primary-foreground rounded text-sm flex items-center gap-1 cursor-pointer"
                                  >
                                    <Check className="w-3 h-3" />
                                    Save
                                  </button>
                                  <button
                                    onClick={cancelEdit}
                                    className="px-3 py-1 bg-muted hover:bg-muted-foreground text-foreground rounded text-sm cursor-pointer"
                                  >
                                    Cancel
                                  </button>
                                </div>
                              </div>
                            ) : (
                              message.content
                            )}
                          </div>

                          {/* Action icons + (arrows if this was the edited user bubble) */}
                          {editingId !== message.id && (
                            <div className="space-y-1 mt-1">
                              {/*copy + edit */}
                              <div className="flex items-center gap-2.5 pt-1 text-xs text-muted-foreground mt-1">
                                <button
                                  onClick={() =>
                                    handleCopy(message.id, message.content)
                                  }
                                  className="hover:text-foreground cursor-pointer"
                                >
                                  {justCopiedId === message.id ? (
                                    <Check className="w-4 h-4" />
                                  ) : (
                                    <Copy className="w-4 h-4" />
                                  )}
                                </button>
                                <button
                                  className="hover:text-foreground cursor-pointer"
                                  onClick={() =>
                                    startEdit(message.id, message.content)
                                  }
                                >
                                  <Edit className="w-4 h-4" />
                                </button>

                                {/*arrows: show arrows if the user message has multiple branches */}
                                {hasMultipleBranches(message.id) && (
                                  <div className="flex items-center gap-1">
                                    <button
                                      onClick={() => goToPrev(message.id)}
                                      disabled={
                                        getBranchIndex(message.id) === 0
                                      }
                                      className={`p-1 rounded cursor-pointer ${
                                        getBranchIndex(message.id) === 0
                                          ? "text-muted-foreground cursor-not-allowed"
                                          : "text-muted-foreground hover:text-foreground hover:bg-accent"
                                      }`}
                                    >
                                      <ArrowLeft className="w-4 h-4" />
                                    </button>
                                    <span className="text-xs text-muted-foreground px-1">
                                      {getBranchIndex(message.id) + 1} /{" "}
                                      {history[currentIndex].branchesByEditId?.[
                                        message.id
                                      ]?.length ?? 1}
                                    </span>
                                    <button
                                      onClick={() => goToNext(message.id)}
                                      disabled={
                                        getBranchIndex(message.id) + 1 ===
                                        (history[currentIndex]
                                          .branchesByEditId?.[message.id]
                                          ?.length ?? 0)
                                      }
                                      className={`p-1 rounded cursor-pointer ${
                                        getBranchIndex(message.id) + 1 ===
                                        (history[currentIndex]
                                          .branchesByEditId?.[message.id]
                                          ?.length ?? 0)
                                          ? "text-muted-foreground cursor-not-allowed"
                                          : "text-muted-foreground hover:text-foreground hover:bg-accent"
                                      }`}
                                    >
                                      <ArrowRight className="w-4 h-4" />
                                    </button>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        // AI bubble
                        message.sender === "ai" && (
                          <div className="max-w-[280px] sm:max-w-xs lg:max-w-md">
                            {message.isThinking ? (
                              // Show only thinking indicator during regeneration
                              <div className="bg-gray-100 text-gray-800 p-3 rounded-2xl rounded-bl-md w-full shadow-sm">
                                <div className="flex items-center gap-2">
                                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
                                  <div
                                    className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"
                                    style={{ animationDelay: "0.2s" }}
                                  ></div>
                                  <div
                                    className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"
                                    style={{ animationDelay: "0.4s" }}
                                  ></div>
                                  <span className="text-gray-600 text-sm ml-2">
                                    Thinking...
                                  </span>
                                </div>
                              </div>
                            ) : (
                              // Show normal message content and buttons
                              <>
                                {(() => {
                                  const html = message.content;
                                  const thinkMatch = html.match(
                                    /<think>([\s\S]*?)<\/think>/i
                                  );
                                  const thoughtHtml = thinkMatch?.[1] ?? "";
                                  const mainHtml = html
                                    .replace(/<think>[\s\S]*?<\/think>/i, "")
                                    .replace(
                                      /###\s*Final\s*Response\s*\n*/gi,
                                      ""
                                    )
                                    .replace(/^\r?\n/, "");
                                  return (
                                    <div>
                                      {/* toggle button */}
                                      {thoughtHtml && (
                                        <button
                                          onClick={() =>
                                            toggleThoughts(message.id)
                                          }
                                          className="text-white mt-2 mr-1 mb-2 text-xs hover:underline cursor-pointer"
                                        >
                                          {showThoughts[message.id]
                                            ? "Hide reasoning"
                                            : "Show reasoning"}
                                        </button>
                                      )}
                                      {/* collapsible reasoning box */}
                                      {thoughtHtml &&
                                        showThoughts[message.id] && (
                                          <div className="bg-blue-50/80 text-gray-800 p-3 rounded-lg mt-1 mb-2 whitespace-pre-wrap border border-blue-200/50">
                                            <div
                                              dangerouslySetInnerHTML={{
                                                __html: thoughtHtml,
                                              }}
                                            />
                                          </div>
                                        )}
                                      {/* final answer */}
                                      <FormattedContent
                                        html={mainHtml}
                                        className="bg-gray-100 text-gray-800 p-3 rounded-2xl rounded-bl-md w-full custom-list max-w-full leading-relaxed shadow-sm"
                                      />
                                    </div>
                                  );
                                })()}
                                {/* Thinking time */}
                                {message.thinkingTime != null && (
                                  <div className="text-muted-foreground text-sm mt-2 pl-1">
                                    Thought for{" "}
                                    {(message.thinkingTime / 1000).toFixed(2)}s
                                  </div>
                                )}

                                {/* Buttons */}
                                <div className="flex gap-2.5 pt-3 pl-1 text-xs text-muted-foreground">
                                  <button
                                    onClick={() =>
                                      handleFeedback(message.id, 0)
                                    }
                                    className="hover:text-foreground cursor-pointer"
                                  >
                                    <ThumbsUp
                                      className={`w-4 h-4 ${
                                        message.feedback === 0
                                          ? "text-green-400"
                                          : "text-muted-foreground hover:text-foreground"
                                      }`}
                                    />
                                  </button>
                                  <button
                                    onClick={() =>
                                      handleFeedback(message.id, 1)
                                    }
                                    className="hover:text-foreground cursor-pointer"
                                  >
                                    <ThumbsDown
                                      className={`w-4 h-4 ${
                                        message.feedback === 1
                                          ? "text-red-400"
                                          : "text-muted-foreground hover:text-foreground"
                                      }`}
                                    />
                                  </button>
                                  <button
                                    onClick={() => handleRegenerate(message.id)}
                                    title={
                                      message.isThinking
                                        ? "Regenerating..."
                                        : "Regenerate response"
                                    }
                                    className={`hover:text-foreground cursor-pointer transition-colors ${
                                      message.isThinking
                                        ? "animate-spin text-blue-500"
                                        : ""
                                    }`}
                                    disabled={
                                      isRegenerating || message.isThinking
                                    }
                                  >
                                    <RefreshCw className="w-4 h-4" />
                                  </button>
                                  <button
                                    onClick={() =>
                                      handleCopy(message.id, message.content)
                                    }
                                    className="hover:text-foreground cursor-pointer"
                                  >
                                    {justCopiedId === message.id ? (
                                      <Check className="w-4 h-4" />
                                    ) : (
                                      <Copy className="w-4 h-4" />
                                    )}
                                  </button>
                                </div>
                              </>
                            )}
                          </div>
                        )
                      )}
                    </div>
                  );
                })}

                {isAwaitingResponse && (
                  <div className="flex gap-3 sm:gap-4">
                    <div className="max-w-[280px] sm:max-w-xs lg:max-w-md">
                      <div className="bg-gray-100 text-gray-800 p-3 rounded-2xl rounded-bl-md w-full shadow-sm">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
                          <div
                            className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"
                            style={{ animationDelay: "0.4s" }}
                          ></div>
                          <span className="text-gray-600 text-sm ml-2">
                            Thinking...
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          )}

          {/* Input Area*/}
          <div className="p-4 sm:p-6 border-t border-border flex-shrink-0">
            <div className="max-w-4xl mx-auto">
              {/* Active Features Display */}
              <div className="mb-3 flex flex-wrap gap-2">
                {/* Engine Type Indicator */}
                <div className="flex items-center gap-2 bg-blue-600 text-white px-3 py-1 rounded-full text-sm">
                  {engineType === "chat" ? (
                    <>
                      <MessageSquare className="w-3 h-3" />
                      <span>Chat Engine</span>
                    </>
                  ) : (
                    <>
                      <Database className="w-3 h-3" />
                      <span>Query Engine</span>
                    </>
                  )}
                </div>

                {/* Other Active Features */}
                {activeFeatures.length > 0 && (
                  <>
                    {activeFeatures.map((feature) => (
                      <div
                        key={feature}
                        className="flex items-center gap-2 bg-primary text-primary-foreground px-3 py-1 rounded-full text-sm"
                      >
                        <span>{feature}</span>
                        <button
                          onClick={() => clearFeature(feature)}
                          className="hover:bg-primary/80 rounded-full p-0.5 transition-colors"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                    <button
                      onClick={clearAllFeatures}
                      className="text-muted-foreground hover:text-foreground text-sm underline transition-colors"
                    >
                      Clear all
                    </button>
                  </>
                )}
              </div>

              <div className="relative">
                <textarea
                  value={inputValue}
                  onChange={(e) => {
                    setInputValue(e.target.value);
                    const ta = e.target;
                    ta.style.height = "auto";
                    ta.style.height = ta.scrollHeight + "px";
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder="Ask Comfit Copilot..."
                  rows={1}
                  className="w-full bg-input border border-border text-foreground px-4 py-3 pb-12 rounded-lg resize-none overflow-hidden"
                  style={{ lineHeight: "1.5", minHeight: "48px" }}
                />
                <div className="absolute inset-x-0 bottom-4 flex justify-between px-2">
                  <div className="flex items-center gap-2">
                    {/* Plus dropdown */}
                    <div className="relative">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          togglePlusDropdown();
                        }}
                        className="flex items-center justify-center w-8 h-8 cursor-pointer hover:bg-accent rounded transition-colors duration-150"
                      >
                        <Plus className="w-5 h-5 text-muted-foreground" />
                      </button>

                      {showPlusDropdown && (
                        <div
                          className="absolute bottom-full left-0 mb-2 w-64 bg-black border border-border rounded-lg shadow-lg z-50"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <div className="p-2">
                            {/* Engine Type Toggle */}
                            <div className="mb-2 p-2 bg-gray-900 rounded">
                              <div className="text-xs text-gray-400 mb-1">
                                Engine Type
                              </div>
                              <div className="flex gap-1">
                                <button
                                  onClick={() => setEngineType("chat")}
                                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                                    engineType === "chat"
                                      ? "bg-blue-600 text-white"
                                      : "text-gray-300 hover:text-white hover:bg-gray-700"
                                  }`}
                                >
                                  <MessageSquare className="w-3 h-3" />
                                  Chat
                                </button>
                                <button
                                  onClick={() => setEngineType("query")}
                                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                                    engineType === "query"
                                      ? "bg-blue-600 text-white"
                                      : "text-gray-300 hover:text-white hover:bg-gray-700"
                                  }`}
                                >
                                  <Database className="w-3 h-3" />
                                  Query
                                </button>
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                {engineType === "chat"
                                  ? "Maintains conversation context"
                                  : "More accurate but no chat context"}
                              </div>
                            </div>

                            {/* Vector Store */}
                            <div className="relative">
                              <button
                                onClick={() => toggleSubmenu("vector-store")}
                                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors cursor-pointer"
                              >
                                <VectorSquare className="w-4 h-4" />
                                <span>Vector Store</span>
                                <ChevronRight className="w-4 h-4 ml-auto" />
                              </button>
                              {openSubmenu === "vector-store" && (
                                <div className="absolute left-full top-0 ml-2 w-64 bg-black border border-border rounded-lg shadow-lg cursor-pointer">
                                  <div className="p-2">
                                    {[
                                      "Vector Store - CFIR",
                                      "Vector Store - Injury Typology",
                                      "Vector Store - Anatomical Regions",
                                    ].map((option) => (
                                      <button
                                        key={option}
                                        onClick={() => selectFeature(option)}
                                        className="w-full flex items-center justify-between px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors cursor-pointer"
                                      >
                                        <span>{option.split(" - ")[1]}</span>
                                        {activeFeatures.includes(option) && (
                                          <Check className="w-4 h-4" />
                                        )}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* Add Files */}
                            <label
                              htmlFor="file-upload"
                              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors cursor-pointer"
                            >
                              <Paperclip className="w-4 h-4" />
                              <span>Add Files</span>
                              <input
                                id="file-upload"
                                type="file"
                                accept=".pdf,.txt"
                                onChange={handleFileChange}
                                disabled={uploading}
                                className="sr-only"
                              />
                            </label>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Voice Mode Icon and send/cancel button */}
                  <div className="flex items-center gap-2">
                    <button className="flex items-center justify-center w-8 h-8 cursor-pointer hover:bg-accent rounded transition-colors duration-150">
                      <Mic className="w-5 h-5 text-muted-foreground" />
                    </button>

                    <button
                      onClick={
                        isAwaitingResponse ? handleCancel : handleSendMessage
                      }
                      disabled={
                        limitReached ||
                        (!isAwaitingResponse && !inputValue.trim())
                      }
                      title={
                        isAwaitingResponse
                          ? "Cancel AI response"
                          : inputValue.trim()
                          ? "Send message"
                          : "Type a message to send"
                      }
                      className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150 ${
                        isAwaitingResponse
                          ? "bg-red-500 hover:bg-red-600 text-white shadow-lg"
                          : inputValue.trim()
                          ? "bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg"
                          : "bg-gray-400 text-gray-600 cursor-not-allowed"
                      }`}
                    >
                      {isAwaitingResponse ? (
                        <X size={20} className="animate-pulse" />
                      ) : (
                        <ArrowUp size={20} />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Sidebar*/}
      {showRightSidebar && (
        <div className="w-64 sm:w-80 bg-sidebar border-l border-sidebar-border flex flex-col max-h-screen overflow-y-auto sidebar-scrollbar">
          <div className="p-3 sm:p-4">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <span className="text-base sm:text-lg font-medium text-sidebar-primary-foreground">
                Advanced Settings
              </span>
              <button
                onClick={() => setShowRightSidebar(false)}
                className="p-1 hover:bg-sidebar-accent rounded cursor-pointer transition-colors"
              >
                <X className="w-4 h-4 cursor-pointer text-sidebar-primary-foreground" />
              </button>
            </div>

            {/* right sidebar accordion */}
            <div className="space-y-2">
              {/* Presets */}
              <div className="border-b border-sidebar-border">
                <button
                  onClick={() => toggleAccordion("presets")}
                  className="w-full text-left text-sm font-medium text-sidebar-primary-foreground hover:text-sidebar-accent-foreground py-3 flex items-center justify-between cursor-pointer transition-colors"
                >
                  Vector Store
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-primary-foreground ${
                      openAccordions.includes("presets") ? "rotate-180" : ""
                    }`}
                  />
                </button>
                {openAccordions.includes("presets") && (
                  <div className="pb-4">
                    <Select
                      value={preset}
                      onValueChange={setPreset}
                      defaultValue="CFIR"
                    >
                      <SelectTrigger className="bg-black-700 border-black">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-black border border-gray-700">
                        <SelectItem value="CFIR">CFIR</SelectItem>
                        <SelectItem value="Head and Neck">
                          Anatomical Regions - Head and Neck
                        </SelectItem>
                        <SelectItem value="Lower Extremity">
                          Anatomical Regions - Lower Extremity
                        </SelectItem>
                        <SelectItem value="Spine">
                          Anatomical Regions - Spine
                        </SelectItem>
                        <SelectItem value="Torso">
                          Anatomical Regions - Torso
                        </SelectItem>
                        <SelectItem value="Upper Extremity">
                          Anatomical Regions - Upper Extremity
                        </SelectItem>
                        <SelectItem value="Fractures">
                          Injury Typology - Fractures
                        </SelectItem>
                        <SelectItem value="Neurological Injuries">
                          Injury Typology - Neurological Injuries
                        </SelectItem>
                        <SelectItem value="Overuse or Chronic Injuries">
                          Injury Typology - Overuse/Chronic Injuries
                        </SelectItem>
                        <SelectItem value="Soft Tissue Injuries">
                          Injury Typology - Soft Tissue Injuries
                        </SelectItem>
                        <SelectItem value="Workplace or Repetitive Strain">
                          Injury Typology - Workplace/Repetitive Strain
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>

              {/* Sampling */}
              <div className="border-b border-sidebar-border">
                <button
                  onClick={() => toggleAccordion("sampling")}
                  className="w-full text-left text-sm font-medium text-sidebar-primary-foreground hover:text-sidebar-accent-foreground py-3 flex items-center justify-between cursor-pointer transition-colors"
                >
                  Sampling
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-primary-foreground ${
                      openAccordions.includes("sampling") ? "rotate-180" : ""
                    }`}
                  />
                </button>
                {openAccordions.includes("sampling") && (
                  <div className="pb-4">
                    <div className="space-y-4">
                      <div>
                        <label className="text-sm text-sidebar-primary-foreground block mb-2">
                          Temperature
                        </label>
                        <input
                          type="number"
                          value={temperature}
                          onChange={(e) =>
                            setTemperature(Number(e.target.value))
                          }
                          step="0.1"
                          min="0"
                          max="2"
                          className="w-full bg-black-700 border border-sidebar-border px-3 py-2 rounded text-sidebar-primary-foreground"
                        />
                      </div>
                      <div>
                        <label className="text-sm text-sidebar-primary-foreground block mb-2">
                          Top P
                        </label>
                        <input
                          type="number"
                          value={topP}
                          onChange={(e) => setTopP(Number(e.target.value))}
                          step="0.1"
                          min="0"
                          max="1"
                          className="w-full bg-black-700 border border-sidebar-border px-3 py-2 rounded text-sidebar-primary-foreground"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Strategies */}
              <div className="border-b border-sidebar-border">
                <button
                  onClick={() => toggleAccordion("strategies")}
                  className="w-full text-left text-sm font-medium text-sidebar-primary-foreground hover:text-sidebar-accent-foreground py-3 flex items-center justify-between cursor-pointer transition-colors"
                >
                  Strategies
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-primary-foreground ${
                      openAccordions.includes("strategies") ? "rotate-180" : ""
                    }`}
                  />
                </button>
                {openAccordions.includes("strategies") && (
                  <div className="pb-4">
                    <Select
                      value={strategy}
                      onValueChange={setStrategy}
                      defaultValue="no-workflow"
                    >
                      <SelectTrigger className="bg-black-700 border-black text-sidebar-primary-foreground">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-black border border-gray-700">
                        <SelectItem value="no-workflow">No workflow</SelectItem>
                        <SelectItem value="ms-query-engine">
                          Multi-Step Query Engine
                        </SelectItem>
                        <SelectItem value="ms-reflection">
                          Multi-Strategy with Reflection
                        </SelectItem>
                        <SelectItem value="query-planning">
                          Query Planning
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>

              {/* Mechanistic Interpretability */}
              <div className="border-b border-sidebar-border">
                <button
                  onClick={() => toggleAccordion("interpretability")}
                  className="w-full text-left text-sm font-medium text-sidebar-primary-foreground hover:text-sidebar-accent-foreground py-3 flex items-center justify-between cursor-pointer transition-colors"
                >
                  <div className="flex items-center gap-2">
                    Mechanistic Interpretability
                  </div>
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-primary-foreground ${
                      openAccordions.includes("interpretability")
                        ? "rotate-180"
                        : ""
                    }`}
                  />
                </button>
                {openAccordions.includes("interpretability") && (
                  <div className="pb-4">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <label className="text-sm text-sidebar-primary-foreground">
                          Enable Interpretability
                        </label>
                        <button
                          onClick={() =>
                            setMechanisticInterpretability(
                              !mechanisticInterpretability
                            )
                          }
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            mechanisticInterpretability
                              ? "bg-primary"
                              : "bg-sidebar-border"
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              mechanisticInterpretability
                                ? "translate-x-6"
                                : "translate-x-1"
                            }`}
                          />
                        </button>
                      </div>
                      {mechanisticInterpretability && (
                        <div className="space-y-3">
                          <div>
                            <label className="text-sm text-sidebar-primary-foreground block mb-2">
                              Interpretability Level
                            </label>
                            <Select defaultValue="basic">
                              <SelectTrigger className="bg-black-700 border-black text-sidebar-primary-foreground">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="bg-black border border-gray-700">
                                <SelectItem value="basic">Basic</SelectItem>
                                <SelectItem value="intermediate">
                                  Intermediate
                                </SelectItem>
                                <SelectItem value="advanced">
                                  Advanced
                                </SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="text-xs text-sidebar-primary-foreground/70">
                            This feature will provide insights into how the AI
                            model processes and generates responses.
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* rate-limit modal*/}
      {showLimitModal && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70">
          <div className="bg-card text-foreground rounded-lg shadow-lg max-w-sm w-full mx-4 p-6">
            <h3 className="text-lg font-semibold mb-2">Rate limit reached</h3>
            <p className="text-sm text-muted-foreground">
              Your message limit has been exceeded, sign up for more messages or
              try again in an hour
            </p>
            <div className="mt-4 flex justify-end space-x-2">
              <button
                onClick={() => router.push("/auth/signup")}
                className="px-4 py-2 bg-primary hover:bg-primary/90 rounded cursor-pointer"
              >
                Sign up
              </button>
              <button
                onClick={() => setShowLimitModal(false)}
                className="px-4 py-2 border border-border rounded hover:bg-accent cursor-pointer"
              >
                Later
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deletingId && (
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/50">
          <div className="bg-gray-800 rounded-lg shadow-lg max-w-sm w-full mx-4">
            <div className="px-6 py-4 border-b border-gray-700">
              <h3 className="text-lg font-semibold text-white">
                Delete Conversation
              </h3>
            </div>
            <div className="px-6 py-4">
              <p className="text-sm text-gray-300">
                Are you sure you want to delete this conversation? This action
                cannot be undone.
              </p>
            </div>
            <div className="px-6 py-4 flex justify-end gap-2 border-t border-gray-700">
              <button
                onClick={() => setDeletingId(null)}
                className="px-4 py-1 text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-700 rounded cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={() => deleteChat(deletingId)}
                className="px-4 py-1 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded cursor-pointer"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* search chats modal*/}
      {showSearchModal && (
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/50">
          <div className="bg-gray-800 rounded-lg shadow-lg w-full max-w-md mx-4 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-foreground">
                Search Chats
              </h3>
              <button
                onClick={() => {
                  setShowSearchModal(false);
                  setSearchQuery("");
                }}
                className="text-muted-foreground hover:text-foreground hover:bg-accent p-1 rounded cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <input
              type="text"
              autoFocus
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Type to search..."
              className="w-full bg-input border border-border text-foreground px-3 py-2 rounded mb-4 focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <div className="max-h-60 overflow-y-auto">
              {filteredResults.length === 0 ? (
                <div className="text-muted-foreground text-sm italic">
                  No matches found
                </div>
              ) : (
                filteredResults.map((conv) => {
                  const idx = conversations.findIndex((c) => c.id === conv.id);
                  return (
                    <div
                      key={conv.id}
                      onClick={() => handleSearchResultClick(idx)}
                      className="cursor-pointer px-3 py-2 rounded hover:bg-accent text-foreground transition-colors"
                    >
                      {conv.title}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
