"use client";

const API_URL = process.env.NEXT_PUBLIC_API_URL;
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

  // states for enhanced Plus dropdown and active features
  const [showPlusDropdown, setShowPlusDropdown] = useState(false);
  const [activeFeatures, setActiveFeatures] = useState<string[]>([]);
  const [openSubmenu, setOpenSubmenu] = useState<string | null>(null);

  // states for editing
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState<string>("");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

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
  const [selectedModel, setSelectedModel] = useState(
    "hf.co/JatinkInnovision/DeKCIB_reasoning_v1:Q4_K_M"
  );
  const [models, setModels] = useState<string[]>([]);
  // right sidebar controls
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful AI assistant for biomechanics and injury"
  );
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [specDecoding, setSpecDecoding] = useState(false);
  const [strategy, setStrategy] = useState("no-workflow");
  const [preset, setPreset] = useState("CFIR");

  // manage profile modal
  const [showProfileModal, setShowProfileModal] = useState(false);

  // AI thinking time
  const [thinkingTime, setThinkingTime] = useState<number | null>(null);

  const openProfileModal = () => {
    setShowUserMenu(false);
    setShowProfileModal(true);
  };
  const closeProfileModal = () => setShowProfileModal(false);

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
      try {
        const res = await fetch(`${API_URL}/api/models`);
        if (!res.ok) throw new Error(await res.text());
        const { models } = await res.json();
        setModels(models);
        if (models.length > 0) setSelectedModel(models[0]);
      } catch (err) {
        console.error("Error loading models:", err);
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
          }),
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
    } catch (err) {
      console.error(err);
    } finally {
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

    setHistory((prev) => {
      const newHist = [...prev];
      newHist[idx] = {
        messages,
        originalMessages: messages,
        branchesByEditId,
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
      })),
      system_prompt: systemPrompt,
      model: selectedModel,
      temperature,
      top_p: topP,
      speculative_decoding: specDecoding,
      strategy,
      preset,
    };

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    const response = await fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      console.error("Failed payload:", payload);
      throw new Error("⚠️ Failed to generate AI response.");
    }

    return await response.json();
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

    const currentMessages = history[currentIndex]?.messages || [];

    // if its a branch
    if (currentBranchId) {
      const messagesForAI = [...currentMessages, userMessage];
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
        ...currentMessages.map((m) => ({
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
    };

    if (session?.access_token) {
      headers.Authorization = `Bearer ${session.access_token}`;
    }
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      // rate limit pop up
      if (res.status === 429) {
        limitReached = true;
        setShowLimitModal(true);
        return;
      }

      if (!res.ok) throw new Error(`Chat API error: ${res.statusText}`);

      const { result: aiText, duration, ai_message } = await res.json();
      setThinkingTime(duration);
      setHistory((prev) => {
        const copy = [...prev];
        copy[currentIndex] = {
          ...copy[currentIndex],
          messages: [
            ...copy[currentIndex].messages,
            {
              id: ai_message.id,
              content: ai_message.content,
              sender: "ai",
              thinkingTime: ai_message.thinking_time,
            },
          ],
          editAtId: undefined,
        };
        return copy;
      });

      setIsAwaitingResponse(false);

      // title generation for first message
      const wasFirst = currentMessages.length === 0;
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
        <div className="w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
          <div className="p-4 border-b border-sidebar-border">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="w-20 h-10 flex items-center justify-center">
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
              className="w-full flex items-center justify-start text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent p-2 rounded cursor-pointer transition-colors"
            >
              <Plus className="w-4 h-4 mr-2" />
              New chat
            </button>
            <button
              onClick={() => setShowSearchModal(true)}
              className="w-full flex items-center justify-start text-sidebar-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent p-2 rounded mt-2 cursor-pointer transition-colors"
            >
              <Search className="w-4 h-4 mr-2" />
              Search chats
            </button>
          </div>

          <div className="flex-1 p-4 overflow-y-auto sidebar-scrollbar">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-sidebar-foreground">
                Recent Conversations
              </span>
            </div>

            {conversations.length === 0 ? (
              <div className="text-sm text-sidebar-foreground italic">
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
                        ? "bg-sidebar-accent text-sidebar-accent-foreground"
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
                      className="text-sm text-sidebar-foreground cursor-pointer"
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
                      <div className="absolute right-0 mt-1 w-32 bg-sidebar-accent border border-sidebar-border rounded shadow-lg z-20">
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

          <div className="p-4 border-t border-sidebar-border relative">
            {user ? (
              <div>
                <button
                  onClick={handleUserMenuToggle}
                  className="flex items-center justify-between gap-2 w-full focus:outline-none"
                >
                  <div className="w-12 h-8 bg-sidebar-accent rounded-full flex items-center justify-center">
                    <UserIcon className="w-4 h-4 text-sidebar-foreground" />
                  </div>
                  <span className="block w-full text-left text-sm text-sidebar-foreground justify-left">
                    {user.user_metadata.full_name}
                  </span>
                  <ChevronDown
                    className={`w-4 h-4 transition-transform text-sidebar-foreground ${
                      showUserMenu ? "rotate-180" : ""
                    }`}
                  />
                </button>

                {showUserMenu && (
                  <div className="absolute bottom-full inset-x-0 ml-2 mb-2 w-9/10 bg-sidebar-accent border border-sidebar-border rounded shadow-lg z-50">
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
                    color-text="black"
                    className="cursor-pointer"
                  >
                    Log in
                  </Button>
                </Link>
                <Link href="/auth/signup">
                  <Button
                    size="lg"
                    variant="default"
                    className="cursor-pointer"
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
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header */}
        <div className="h-16 border-b border-border flex items-center justify-between px-6 flex-shrink-0">
          <div className="flex items-center gap-4">
            {!showLeftSidebar && (
              <button
                onClick={() => setShowLeftSidebar(true)}
                className="text-muted-foreground hover:text-foreground hover:bg-accent p-2 rounded"
              >
                <Menu className="w-4 h-4" />
              </button>
            )}
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="w-48 bg-input border-border">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
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

          {!showRightSidebar && (
            <button
              onClick={() => setShowRightSidebar(!showRightSidebar)}
              className="text-muted-foreground hover:text-foreground hover:bg-accent p-2 rounded flex items-center gap-2"
            >
              <Settings className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-h-0">
          {isWelcomeState ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8">
              <h1 className="text-3xl font-light mb-12 text-center">
                What would you like to know?
              </h1>

              <div className="flex gap-4 mb-8">
                <button
                  className="border border-border text-muted-foreground hover:bg-accent px-4 py-2 rounded cursor-pointer"
                  onClick={() =>
                    handleExampleClick(
                      "What is your name and who developed you?"
                    )
                  }
                >
                  What is your name and who developed you?
                </button>
                <button
                  className="border border-border text-muted-foreground hover:bg-accent px-4 py-2 rounded cursor-pointer"
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
                  className="border border-border text-muted-foreground hover:bg-accent px-4 py-2 rounded cursor-pointer"
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
            <div className="flex-1 overflow-y-auto p-6 chat-scrollbar">
              <div className="max-w-3xl mx-auto space-y-6">
                {currentMessages.map((message, idx) => {
                  return (
                    <div key={message.id} className="flex gap-4">
                      {message.sender === "user" ? (
                        <div className="ml-auto max-w-xs lg:max-w-md flex flex-col items-end">
                          {/* User bubble */}
                          <div className="bg-secondary text-secondary-foreground p-2.5 rounded-lg whitespace-pre-wrap break-words max-w-full">
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
                          <div className="max-w-xs lg:max-w-md">
                            {message.isThinking ? (
                              // Show only thinking indicator during regeneration
                              <div className="bg-card text-card-foreground p-3 rounded-lg w-full">
                                <div className="flex items-center gap-2">
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse"></div>
                                  <div
                                    className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse"
                                    style={{ animationDelay: "0.2s" }}
                                  ></div>
                                  <div
                                    className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse"
                                    style={{ animationDelay: "0.4s" }}
                                  ></div>
                                  <span className="text-muted-foreground text-sm ml-2">
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
                                          className="mt-2 mr-1 mb-2 text-xs text-primary hover:underline cursor-pointer"
                                        >
                                          {showThoughts[message.id]
                                            ? "Hide reasoning"
                                            : "Show reasoning"}
                                        </button>
                                      )}
                                      {/* collapsible reasoning box */}
                                      {thoughtHtml &&
                                        showThoughts[message.id] && (
                                          <div className="bg-card text-card-foreground p-3 rounded-lg mt-1 mb-2 whitespace-pre-wrap">
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
                                        className="bg-card text-card-foreground p-3 rounded-lg w-full custom-list max-w-full leading-relaxed"
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
                                    className="hover:text-foreground cursor-pointer"
                                    disabled={isRegenerating}
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
                  <div className="flex gap-4">
                    <div className="max-w-xs lg:max-w-md">
                      <div className="bg-card text-card-foreground p-3 rounded-lg w-full">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse"></div>
                          <div
                            className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse"
                            style={{ animationDelay: "0.4s" }}
                          ></div>
                          <span className="text-muted-foreground text-sm ml-2">
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
          <div className="p-6 border-t border-border flex-shrink-0">
            <div className="max-w-3xl mx-auto">
              {/* Active Features Display */}
              {activeFeatures.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2">
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
                </div>
              )}

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
                            {/* Vector Store */}
                            <div className="relative">
                              <button
                                onClick={() => toggleSubmenu("vector-store")}
                                className="w-full flex items-center justify-between px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors"
                              >
                                <span>Vector Store</span>
                                <ChevronRight
                                  className={`w-4 h-4 transition-transform ${
                                    openSubmenu === "vector-store"
                                      ? "rotate-90"
                                      : ""
                                  }`}
                                />
                              </button>
                              {openSubmenu === "vector-store" && (
                                <div className="ml-4 mt-1 space-y-1">
                                  {[
                                    "Vector Store - Medical",
                                    "Vector Store - Research",
                                    "Vector Store - Clinical",
                                  ].map((option) => (
                                    <button
                                      key={option}
                                      onClick={() => selectFeature(option)}
                                      className="w-full flex bg-black items-center justify-between px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors"
                                    >
                                      <span>{option.split(" - ")[1]}</span>
                                      {activeFeatures.includes(option) && (
                                        <Check className="w-4 h-4" />
                                      )}
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>

                            {/* Upload File */}
                            <label
                              htmlFor="file-upload"
                              className="w-full flex items-center px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors cursor-pointer"
                            >
                              <span>Upload File</span>
                              <input
                                id="file-upload"
                                type="file"
                                accept=".pdf,.txt"
                                onChange={handleFileChange}
                                disabled={uploading}
                                className="sr-only"
                              />
                            </label>

                            {/* Web Search */}
                            <button
                              onClick={() => selectFeature("Web Search")}
                              className="w-full flex items-center justify-between px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors"
                            >
                              <span>Web Search</span>
                              {activeFeatures.includes("Web Search") && (
                                <Check className="w-4 h-4" />
                              )}
                            </button>

                            {/* Strategies */}
                            <div className="relative">
                              <button
                                onClick={() => toggleSubmenu("strategies")}
                                className="w-full flex items-center justify-between px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors"
                              >
                                <span>Strategies</span>
                                <ChevronRight
                                  className={`w-4 h-4 transition-transform ${
                                    openSubmenu === "strategies"
                                      ? "rotate-90"
                                      : ""
                                  }`}
                                />
                              </button>
                              {openSubmenu === "strategies" && (
                                <div className="ml-4 mt-1 space-y-1">
                                  {[
                                    "Strategies - Reasoning",
                                    "Strategies - Analysis",
                                    "Strategies - Synthesis",
                                  ].map((option) => (
                                    <button
                                      key={option}
                                      onClick={() => selectFeature(option)}
                                      className="w-full flex items-center justify-between px-3 py-2 text-sm text-foreground hover:bg-accent rounded transition-colors"
                                    >
                                      <span>{option.split(" - ")[1]}</span>
                                      {activeFeatures.includes(option) && (
                                        <Check className="w-4 h-4" />
                                      )}
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Voice Mode Icon - needs implementation*/}
                    <button className="flex items-center justify-center w-8 h-8 cursor-pointer hover:bg-accent rounded transition-colors duration-150">
                      <Mic className="w-5 h-5 text-muted-foreground" />
                    </button>
                  </div>

                  {/* send/cancel button */}
                  <button
                    onClick={
                      isAwaitingResponse ? handleCancel : handleSendMessage
                    }
                    disabled={
                      limitReached ||
                      (!isAwaitingResponse && !inputValue.trim())
                    }
                    className="w-8 h-8 rounded-lg bg-primary hover:bg-primary/80 flex items-center justify-center transition-colors duration-150 disabled:bg-primary/60 disabled:cursor-not-allowed"
                  >
                    {isAwaitingResponse ? (
                      <X size={20} />
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

      {/* Right Sidebar*/}
      {showRightSidebar && (
        <div className="w-80 bg-sidebar border-l border-sidebar-border flex flex-col max-h-screen overflow-y-auto sidebar-scrollbar">
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <span className="text-lg font-medium text-sidebar-primary-foreground">
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
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/70">
          <div className="bg-card rounded-lg shadow-lg max-w-sm w-full mx-4">
            <div className="px-6 py-4 border-b border-border">
              <h3 className="text-lg font-semibold text-foreground">
                Delete Conversation
              </h3>
            </div>
            <div className="px-6 py-4">
              <p className="text-sm text-muted-foreground">
                Are you sure you want to delete this conversation? This action
                cannot be undone.
              </p>
            </div>
            <div className="px-6 py-4 flex justify-end gap-2 border-t border-border">
              <button
                onClick={() => setDeletingId(null)}
                className="px-4 py-1 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent rounded cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={() => deleteChat(deletingId)}
                className="px-4 py-1 text-sm font-medium text-white bg-destructive hover:bg-destructive/90 rounded cursor-pointer"
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
          <div className="bg-card rounded-lg shadow-lg w-full max-w-md mx-4 p-6">
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
