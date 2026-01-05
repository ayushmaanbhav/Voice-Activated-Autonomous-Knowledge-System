/**
 * Simple Push-to-Talk Interface
 *
 * A minimal UI for testing the STT -> LLM -> TTS pipeline without VAD/turn detection.
 * User holds to record, releases to process.
 *
 * Flow:
 * 1. User clicks and holds mic button -> starts recording
 * 2. User releases -> sends audio to backend (or can discard)
 * 3. Backend: STT -> Translation -> RAG/LLM -> Reverse Translation -> TTS
 * 4. Play TTS audio back
 */

import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import '../styles/animations.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  originalText?: string;
  timestamp: Date;
}

type ProcessingStage = 'idle' | 'recording' | 'processing' | 'stt' | 'llm' | 'tts' | 'playing';

interface ProcessingState {
  stage: ProcessingStage;
  message?: string;
}

// Encouraging phrases for different stages
const ENCOURAGING_PHRASES: Record<ProcessingStage, string[]> = {
  idle: [],
  recording: [
    'Listening...',
    'Go ahead, I\'m here...',
    'Speak your mind...',
    'I\'m all ears...',
    'Recording your voice...',
  ],
  processing: [
    'Processing your request...',
    'Working on it...',
    'Just a moment...',
    'Analyzing...',
    'Thinking...',
  ],
  stt: [
    'Understanding your words...',
    'Converting speech to text...',
    'Transcribing...',
    'Listening carefully...',
    'Decoding your message...',
  ],
  llm: [
    'Crafting a response...',
    'Thinking deeply...',
    'Finding the best answer...',
    'Almost there...',
    'Preparing your answer...',
  ],
  tts: [
    'Preparing voice response...',
    'Getting ready to speak...',
    'Converting to speech...',
    'Finding my voice...',
    'Preparing audio...',
  ],
  playing: [
    'Speaking...',
    'Here\'s what I found...',
    'Listen up...',
    'Playing response...',
    'Here you go...',
  ],
};

// Backend API configuration
const API_BASE = '/api/ptt';

// Get random phrase from list
const getRandomPhrase = (stage: ProcessingStage): string => {
  const phrases = ENCOURAGING_PHRASES[stage];
  if (!phrases) return '';
  return phrases[Math.floor(Math.random() * phrases.length)];
};

export default function SimplePTT() {
  // State
  const [messages, setMessages] = useState<Message[]>([]);
  const [processingState, setProcessingState] = useState<ProcessingState>({ stage: 'idle' });
  const [currentPhrase, setCurrentPhrase] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<string>('en');
  const [prevLanguage, setPrevLanguage] = useState<string>('en');
  const [isLoadingGreeting, setIsLoadingGreeting] = useState(true);
  const [isTranslating, setIsTranslating] = useState(false);

  // Use ref to access current messages in async functions without stale closure
  const messagesRef = useRef<Message[]>([]);
  messagesRef.current = messages;

  // Translate messages when language changes
  const translateMessages = useCallback(async (fromLang: string, toLang: string, currentMessages: Message[]): Promise<Message[]> => {
    if (currentMessages.length === 0 || fromLang === toLang) return currentMessages;

    setIsTranslating(true);
    try {
      // Prepare messages for translation (skip greeting since we'll get new one)
      const messagesToTranslate = currentMessages
        .filter(msg => !msg.id.includes('greeting'))
        .map((msg) => ({
          id: msg.id,
          text: msg.text,
          role: msg.role,
        }));

      if (messagesToTranslate.length === 0) {
        setIsTranslating(false);
        return currentMessages;
      }

      const response = await fetch(`${API_BASE}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messagesToTranslate,
          source_language: fromLang,
          target_language: toLang,
        }),
      });

      if (response.ok) {
        const data = await response.json();

        // Update messages with translations
        const result = currentMessages.map((msg) => {
          if (msg.id.includes('greeting')) return msg; // Keep greeting, it'll be replaced
          const translated = data.messages.find((t: { id: string; text: string; original: string }) => t.id === msg.id);
          if (translated) {
            return {
              ...msg,
              text: translated.text,
              originalText: translated.original !== translated.text ? translated.original : msg.originalText,
            };
          }
          return msg;
        });
        setIsTranslating(false);
        return result;
      }
    } catch (err) {
      console.error('Translation failed:', err);
    }
    setIsTranslating(false);
    return currentMessages;
  }, []);

  // Fetch greeting from backend
  const fetchGreetingOnly = useCallback(async (lang: string): Promise<Message> => {
    try {
      const response = await fetch(`${API_BASE}/greeting`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: lang }),
      });

      if (response.ok) {
        const data = await response.json();
        return {
          id: `assistant-greeting-${Date.now()}`,
          role: 'assistant',
          text: data.greeting,
          originalText: data.greeting_english !== data.greeting ? data.greeting_english : undefined,
          timestamp: new Date(),
        };
      }
    } catch (err) {
      console.error('Greeting fetch failed:', err);
    }
    // Fallback greeting
    return {
      id: `assistant-greeting-${Date.now()}`,
      role: 'assistant',
      text: "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?",
      timestamp: new Date(),
    };
  }, []);

  // Handle language change: translate existing messages and get new greeting
  useEffect(() => {
    const handleLanguageChange = async () => {
      setIsLoadingGreeting(true);

      // Get new greeting for the new language
      const newGreeting = await fetchGreetingOnly(language);

      const currentMessages = messagesRef.current;
      if (currentMessages.length <= 1 || language === prevLanguage) {
        // First load or same language: just set the greeting
        setMessages([newGreeting]);
      } else {
        // Translate existing conversation messages (except greeting)
        const translatedMessages = await translateMessages(prevLanguage, language, currentMessages);

        // Replace old greeting with new one, keep translated conversation
        const conversationMessages = translatedMessages.filter(msg => !msg.id.includes('greeting'));
        setMessages([newGreeting, ...conversationMessages]);
      }

      setPrevLanguage(language);
      setIsLoadingGreeting(false);
    };

    handleLanguageChange();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language]); // Only depend on language to avoid infinite loops
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [showDiscardOption, setShowDiscardOption] = useState(false);
  const [autoSendCountdown, setAutoSendCountdown] = useState<number>(3);

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const autoSendTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoSendTriggeredRef = useRef<boolean>(false);
  const sendRecordingRef = useRef<() => Promise<void>>();

  // Generate stable wave bar heights for voice message preview
  const waveBarHeights = useMemo(() =>
    [...Array(20)].map(() => Math.random() * 16 + 4),
  []);

  // Update phrase when stage changes
  useEffect(() => {
    if (processingState.stage !== 'idle') {
      setCurrentPhrase(getRandomPhrase(processingState.stage));

      // Rotate phrases every 2.5 seconds for longer processing stages
      const interval = setInterval(() => {
        if (processingState.stage !== 'idle' && processingState.stage !== 'recording') {
          setCurrentPhrase(getRandomPhrase(processingState.stage));
        }
      }, 2500);

      return () => clearInterval(interval);
    }
  }, [processingState.stage]);

  // Auto-send countdown timer
  useEffect(() => {
    if (showDiscardOption) {
      setAutoSendCountdown(3);
      autoSendTriggeredRef.current = false; // Reset on new recording

      autoSendTimerRef.current = setInterval(() => {
        setAutoSendCountdown(prev => {
          if (prev <= 1) {
            // Clear interval first to prevent any race conditions
            if (autoSendTimerRef.current) {
              clearInterval(autoSendTimerRef.current);
              autoSendTimerRef.current = null;
            }
            // Trigger send only once using ref guard
            if (!autoSendTriggeredRef.current) {
              autoSendTriggeredRef.current = true;
              setTimeout(() => sendRecordingRef.current?.(), 100);
            }
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => {
        if (autoSendTimerRef.current) {
          clearInterval(autoSendTimerRef.current);
          autoSendTimerRef.current = null;
        }
      };
    }
  }, [showDiscardOption]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, []);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setIsRecording(true);
      setShowDiscardOption(false);
      setRecordingDuration(0);
      setProcessingState({ stage: 'recording' });
      audioChunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });
      streamRef.current = stream;

      // Set up audio analysis for visualization
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);

      // Start level monitoring
      const updateLevel = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          setAudioLevel(average / 255);
          animationFrameRef.current = requestAnimationFrame(updateLevel);
        }
      };
      updateLevel();

      // Start recording duration timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 100);
      }, 100);

      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100);

    } catch (err) {
      setError(`Failed to access microphone: ${err}`);
      setIsRecording(false);
      setProcessingState({ stage: 'idle' });
    }
  }, []);

  // Stop recording (shows discard option)
  const stopRecording = useCallback(() => {
    if (!mediaRecorderRef.current || !isRecording) return;

    setIsRecording(false);
    setShowDiscardOption(true);

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
    }
    setAudioLevel(0);

    // Stop the media recorder but keep the data
    mediaRecorderRef.current.stop();
  }, [isRecording]);

  // Discard recording
  const discardRecording = useCallback(() => {
    // Clear auto-send timer and reset flag
    if (autoSendTimerRef.current) {
      clearInterval(autoSendTimerRef.current);
      autoSendTimerRef.current = null;
    }
    autoSendTriggeredRef.current = false;

    audioChunksRef.current = [];
    setShowDiscardOption(false);
    setProcessingState({ stage: 'idle' });
    setRecordingDuration(0);

    // Stop stream tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
  }, []);

  // Send recording
  const sendRecording = useCallback(async () => {
    // Clear auto-send timer and reset flag
    if (autoSendTimerRef.current) {
      clearInterval(autoSendTimerRef.current);
      autoSendTimerRef.current = null;
    }
    autoSendTriggeredRef.current = false;

    setShowDiscardOption(false);

    // Stop stream tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }

    // Create blob from chunks
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

    if (audioBlob.size < 1000) {
      setError('Recording too short');
      setProcessingState({ stage: 'idle' });
      return;
    }

    // Process the audio
    await processAudio(audioBlob);
  }, [language]);

  // Keep ref in sync with sendRecording callback
  sendRecordingRef.current = sendRecording;

  // Process audio through backend
  const processAudio = async (audioBlob: Blob) => {
    try {
      setProcessingState({ stage: 'processing' });

      // Convert blob to base64 for sending
      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

      // Send to backend
      setProcessingState({ stage: 'stt' });

      const response = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio: base64Audio,
          audio_format: 'webm',
          language: language,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend error: ${response.status} - ${errorText}`);
      }

      setProcessingState({ stage: 'llm' });
      const result = await response.json();

      // Add user message
      if (result.user_text) {
        setMessages(prev => [...prev, {
          id: `user-${Date.now()}`,
          role: 'user',
          text: result.user_text,
          originalText: result.user_text_original,
          timestamp: new Date(),
        }]);
      }

      // Add assistant message
      if (result.assistant_text) {
        setMessages(prev => [...prev, {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          text: result.assistant_text,
          originalText: result.assistant_text_original,
          timestamp: new Date(),
        }]);
      }

      // Play TTS audio if available
      if (result.audio_response) {
        setProcessingState({ stage: 'playing' });
        await playAudio(result.audio_response, result.audio_format || 'wav');
      }

      setProcessingState({ stage: 'idle' });

    } catch (err) {
      setError(`Processing failed: ${err}`);
      setProcessingState({ stage: 'idle' });
    }
  };

  // Play audio response
  const playAudio = async (base64Audio: string, format: string) => {
    return new Promise<void>((resolve) => {
      const audioData = atob(base64Audio);
      const audioArray = new Uint8Array(audioData.length);
      for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i);
      }

      const blob = new Blob([audioArray], { type: `audio/${format}` });
      const url = URL.createObjectURL(blob);

      if (audioElementRef.current) {
        audioElementRef.current.pause();
      }

      const audio = new Audio(url);
      audioElementRef.current = audio;

      audio.onended = () => {
        URL.revokeObjectURL(url);
        resolve();
      };

      audio.onerror = () => {
        URL.revokeObjectURL(url);
        resolve();
      };

      audio.play().catch(() => resolve());
    });
  };

  // Handle mouse/touch events for push-to-talk
  const handlePressStart = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    if (processingState.stage === 'idle') {
      startRecording();
    }
  }, [processingState.stage, startRecording]);

  const handlePressEnd = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    if (isRecording) {
      stopRecording();
    }
  }, [isRecording, stopRecording]);

  // Format duration
  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const tenths = Math.floor((ms % 1000) / 100);
    return `${seconds}.${tenths}s`;
  };

  // Get status color based on stage
  const getStatusColor = (stage: ProcessingStage) => {
    switch (stage) {
      case 'recording': return '#ef4444';
      case 'processing':
      case 'stt':
      case 'llm':
      case 'tts': return '#fbbf24';
      case 'playing': return '#4ade80';
      default: return '#60a5fa';
    }
  };

  // Check if processing (not idle or recording)
  const isProcessing = processingState.stage !== 'idle' && processingState.stage !== 'recording' && !showDiscardOption;

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>Voice Agent - Simple Mode</h1>
        <div style={styles.headerControls}>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            style={styles.select}
          >
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="ta">Tamil</option>
            <option value="te">Telugu</option>
            <option value="kn">Kannada</option>
            <option value="ml">Malayalam</option>
          </select>
          <a href="#/" style={styles.backLink}>← Full UI</a>
        </div>
      </header>

      {/* Messages */}
      <div style={styles.messagesContainer}>
        {isLoadingGreeting ? (
          <div style={styles.emptyState}>
            <div style={styles.loadingGreeting}>
              <div style={styles.loaderDots}>
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    style={{
                      ...styles.loaderDot,
                      animation: 'bounce 1.4s ease-in-out infinite',
                      animationDelay: `${i * 0.16}s`,
                      background: '#60a5fa',
                    }}
                  />
                ))}
              </div>
              <p style={styles.loadingText}>{isTranslating ? 'Translating messages...' : 'Loading assistant...'}</p>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div style={styles.emptyState}>
            <div style={styles.emptyIcon}>
              <MicIconLarge />
            </div>
            <p style={styles.emptyTitle}>Hold the mic button to speak</p>
            <p style={styles.hint}>Release to send • Swipe away to discard</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={msg.id}
              style={{
                ...styles.message,
                ...(msg.role === 'user' ? styles.userMessage : styles.assistantMessage),
                animation: 'fadeSlideUp 0.3s ease-out',
                animationDelay: `${index * 0.05}s`,
              }}
            >
              <div style={styles.messageRole}>
                {msg.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div style={styles.messageText}>{msg.text}</div>
              {msg.originalText && msg.originalText !== msg.text && (
                <div style={styles.originalText}>
                  Original: {msg.originalText}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Status Indicator */}
      {(processingState.stage !== 'idle' || showDiscardOption) && (
        <div style={{
          ...styles.statusContainer,
          animation: 'fadeSlideUp 0.2s ease-out',
        }}>
          {showDiscardOption ? (
            <div style={styles.voiceMessageBar}>
              {/* Discard button */}
              <button
                style={styles.discardButtonCompact}
                onClick={discardRecording}
                title="Discard"
              >
                <TrashIcon />
              </button>

              {/* Voice message preview */}
              <div style={styles.voiceMessagePreview}>
                <WaveformIcon />
                <div style={styles.voiceMessageWave}>
                  {waveBarHeights.map((height, i) => (
                    <div
                      key={i}
                      style={{
                        ...styles.waveBarStatic,
                        height: `${height}px`,
                      }}
                    />
                  ))}
                </div>
                <span style={styles.durationTextCompact}>{formatDuration(recordingDuration)}</span>
              </div>

              {/* Send button with countdown */}
              <button
                style={styles.sendButtonWithTimer}
                onClick={sendRecording}
              >
                <div style={styles.countdownContainer}>
                  <svg width="44" height="44" viewBox="0 0 44 44" style={styles.countdownRing}>
                    <circle
                      cx="22"
                      cy="22"
                      r="18"
                      fill="none"
                      stroke="rgba(255,255,255,0.2)"
                      strokeWidth="3"
                    />
                    <circle
                      cx="22"
                      cy="22"
                      r="18"
                      fill="none"
                      stroke="#fff"
                      strokeWidth="3"
                      strokeLinecap="round"
                      strokeDasharray={`${2 * Math.PI * 18}`}
                      strokeDashoffset={`${2 * Math.PI * 18 * (1 - autoSendCountdown / 3)}`}
                      style={{
                        transition: 'stroke-dashoffset 0.3s ease-out',
                        transform: 'rotate(-90deg)',
                        transformOrigin: 'center',
                      }}
                    />
                  </svg>
                  <span
                    key={autoSendCountdown}
                    style={{
                      ...styles.countdownNumber,
                      animation: 'countdownPop 0.3s ease-out',
                    }}
                  >
                    {autoSendCountdown}
                  </span>
                </div>
              </button>
            </div>
          ) : (
            <div style={styles.statusInner}>
              {isRecording && (
                <div style={styles.recordingStatus}>
                  <div style={{
                    ...styles.recordingDot,
                    animation: 'statusPulse 1s ease-in-out infinite',
                  }} />
                  <span style={styles.recordingTime}>{formatDuration(recordingDuration)}</span>
                </div>
              )}

              {isProcessing && (
                <ProcessingLoader stage={processingState.stage} />
              )}

              <div style={{
                ...styles.statusText,
                color: getStatusColor(processingState.stage),
              }}>
                {currentPhrase}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          ...styles.error,
          animation: 'shake 0.5s ease-in-out',
        }}>
          {error}
          <button onClick={() => setError(null)} style={styles.dismissButton}>×</button>
        </div>
      )}

      {/* Mic Button Area */}
      <div style={styles.micContainer}>
        <div style={styles.micButtonWrapper}>
          {/* Pulse rings when recording */}
          {isRecording && (
            <>
              <div style={{
                ...styles.pulseRing,
                animation: 'pulseRing 1.5s ease-out infinite',
              }} />
              <div style={{
                ...styles.pulseRing,
                animation: 'pulseRing 1.5s ease-out infinite 0.5s',
              }} />
              <div style={{
                ...styles.pulseRing,
                animation: 'pulseRing 1.5s ease-out infinite 1s',
              }} />
            </>
          )}

          {/* Processing orbit animation */}
          {isProcessing && (
            <div style={styles.orbitContainer}>
              <div style={{
                ...styles.orbitDot,
                animation: 'orbit 1.5s linear infinite',
              }} />
              <div style={{
                ...styles.orbitDot,
                ...styles.orbitDot2,
                animation: 'orbit 1.5s linear infinite 0.5s',
              }} />
              <div style={{
                ...styles.orbitDot,
                ...styles.orbitDot3,
                animation: 'orbit 1.5s linear infinite 1s',
              }} />
            </div>
          )}

          <button
            style={{
              ...styles.micButton,
              ...(isRecording ? styles.micButtonRecording : {}),
              ...(isProcessing ? styles.micButtonProcessing : {}),
              ...(processingState.stage === 'playing' ? styles.micButtonPlaying : {}),
              transform: isRecording ? `scale(${1 + audioLevel * 0.2})` : 'scale(1)',
            }}
            onMouseDown={handlePressStart}
            onMouseUp={handlePressEnd}
            onMouseLeave={handlePressEnd}
            onTouchStart={handlePressStart}
            onTouchEnd={handlePressEnd}
            disabled={processingState.stage !== 'idle' && processingState.stage !== 'recording'}
          >
            {isProcessing ? (
              <ProcessingIcon stage={processingState.stage} />
            ) : processingState.stage === 'playing' ? (
              <SpeakerIcon />
            ) : (
              <MicIcon isRecording={isRecording} />
            )}
          </button>
        </div>

        <p style={styles.micHint}>
          {showDiscardOption ? 'Choose an action above' :
           isRecording ? 'Release to preview' :
           isProcessing ? 'Processing...' :
           processingState.stage === 'playing' ? 'Playing response...' :
           'Hold to speak'}
        </p>
      </div>
    </div>
  );
}

// Processing Loader Component
function ProcessingLoader({ stage }: { stage: ProcessingStage }) {
  return (
    <div style={styles.processingLoader}>
      <div style={styles.loaderDots}>
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            style={{
              ...styles.loaderDot,
              animation: 'bounce 1.4s ease-in-out infinite',
              animationDelay: `${i * 0.16}s`,
              background: stage === 'playing' ? '#4ade80' : '#fbbf24',
            }}
          />
        ))}
      </div>
    </div>
  );
}

// Processing Icon Component
function ProcessingIcon({ stage }: { stage: ProcessingStage }) {
  const color = stage === 'playing' ? '#4ade80' : '#fbbf24';
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
      <circle cx="12" cy="12" r="10" strokeDasharray="60" strokeDashoffset="0">
        <animate
          attributeName="stroke-dashoffset"
          values="0;60"
          dur="1.5s"
          repeatCount="indefinite"
        />
      </circle>
      <path d="M12 6v6l4 2" />
    </svg>
  );
}

// Mic Icon Component
function MicIcon({ isRecording }: { isRecording: boolean }) {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke={isRecording ? '#fff' : '#fff'} strokeWidth="2">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

// Large Mic Icon for empty state
function MicIconLarge() {
  return (
    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="1.5">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

// Speaker Icon for playing state
function SpeakerIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07">
        <animate
          attributeName="opacity"
          values="1;0.5;1"
          dur="0.8s"
          repeatCount="indefinite"
        />
      </path>
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14">
        <animate
          attributeName="opacity"
          values="1;0.3;1"
          dur="0.8s"
          repeatCount="indefinite"
          begin="0.2s"
        />
      </path>
    </svg>
  );
}

// Waveform Icon for recording preview
function WaveformIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="#60a5fa">
      <rect x="2" y="10" width="2" height="4" rx="1" style={{ animation: 'waveBar 0.5s ease-in-out infinite', transformOrigin: 'center' }} />
      <rect x="6" y="7" width="2" height="10" rx="1" style={{ animation: 'waveBar 0.6s ease-in-out infinite 0.1s', transformOrigin: 'center' }} />
      <rect x="10" y="4" width="2" height="16" rx="1" style={{ animation: 'waveBar 0.5s ease-in-out infinite 0.2s', transformOrigin: 'center' }} />
      <rect x="14" y="7" width="2" height="10" rx="1" style={{ animation: 'waveBar 0.6s ease-in-out infinite 0.3s', transformOrigin: 'center' }} />
      <rect x="18" y="10" width="2" height="4" rx="1" style={{ animation: 'waveBar 0.5s ease-in-out infinite 0.4s', transformOrigin: 'center' }} />
    </svg>
  );
}

// Trash Icon
function TrashIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}


const styles: Record<string, React.CSSProperties> = {
  container: {
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: 'linear-gradient(180deg, #0f172a 0%, #1e1e3f 100%)',
    color: '#e2e8f0',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 24px',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
    background: 'rgba(30, 41, 59, 0.8)',
    backdropFilter: 'blur(10px)',
  },
  title: {
    fontSize: '20px',
    fontWeight: 600,
    margin: 0,
    background: 'linear-gradient(90deg, #60a5fa, #a78bfa)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  headerControls: {
    display: 'flex',
    gap: '16px',
    alignItems: 'center',
  },
  select: {
    padding: '8px 12px',
    borderRadius: '8px',
    border: '1px solid rgba(255,255,255,0.2)',
    background: 'rgba(15, 23, 42, 0.8)',
    color: '#e2e8f0',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'border-color 0.2s',
  },
  backLink: {
    color: '#60a5fa',
    textDecoration: 'none',
    fontSize: '14px',
    transition: 'opacity 0.2s',
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto',
    padding: '24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  emptyState: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    color: '#64748b',
    textAlign: 'center',
  },
  emptyIcon: {
    marginBottom: '24px',
    opacity: 0.5,
  },
  loadingGreeting: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '16px',
  },
  loadingText: {
    fontSize: '16px',
    color: '#94a3b8',
  },
  emptyTitle: {
    fontSize: '20px',
    fontWeight: 500,
    color: '#94a3b8',
    marginBottom: '8px',
  },
  hint: {
    fontSize: '14px',
    color: '#64748b',
  },
  message: {
    maxWidth: '80%',
    padding: '14px 18px',
    borderRadius: '16px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
  },
  userMessage: {
    alignSelf: 'flex-end',
    background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
    borderBottomRightRadius: '4px',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    background: 'linear-gradient(135deg, #334155, #1e293b)',
    borderBottomLeftRadius: '4px',
  },
  messageRole: {
    fontSize: '11px',
    opacity: 0.7,
    marginBottom: '4px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  messageText: {
    fontSize: '16px',
    lineHeight: 1.5,
  },
  originalText: {
    fontSize: '12px',
    opacity: 0.6,
    marginTop: '8px',
    fontStyle: 'italic',
    borderTop: '1px solid rgba(255,255,255,0.1)',
    paddingTop: '8px',
  },
  statusContainer: {
    padding: '16px 24px',
    background: 'rgba(30, 41, 59, 0.9)',
    backdropFilter: 'blur(10px)',
    borderTop: '1px solid rgba(255,255,255,0.1)',
  },
  statusInner: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '16px',
  },
  statusText: {
    fontSize: '15px',
    fontWeight: 500,
    transition: 'color 0.3s',
  },
  recordingStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  recordingDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    background: '#ef4444',
  },
  recordingTime: {
    fontSize: '14px',
    fontFamily: 'monospace',
    color: '#ef4444',
  },
  voiceMessageBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '8px 12px',
    background: 'rgba(30, 41, 59, 0.95)',
    borderRadius: '28px',
    border: '1px solid rgba(96, 165, 250, 0.3)',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
  },
  discardButtonCompact: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    border: 'none',
    background: 'rgba(239, 68, 68, 0.15)',
    color: '#ef4444',
    cursor: 'pointer',
    transition: 'all 0.2s',
    flexShrink: 0,
  },
  voiceMessagePreview: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flex: 1,
    padding: '4px 8px',
  },
  voiceMessageWave: {
    display: 'flex',
    alignItems: 'center',
    gap: '2px',
    flex: 1,
    height: '24px',
  },
  waveBarStatic: {
    width: '3px',
    background: 'linear-gradient(180deg, #60a5fa, #3b82f6)',
    borderRadius: '2px',
    opacity: 0.7,
  },
  durationTextCompact: {
    fontSize: '13px',
    fontFamily: 'monospace',
    color: '#94a3b8',
    flexShrink: 0,
  },
  sendButtonWithTimer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    border: 'none',
    background: 'linear-gradient(135deg, #22c55e, #16a34a)',
    color: '#fff',
    cursor: 'pointer',
    transition: 'all 0.2s',
    boxShadow: '0 4px 12px rgba(34, 197, 94, 0.4)',
    flexShrink: 0,
    position: 'relative',
  },
  countdownContainer: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '44px',
    height: '44px',
  },
  countdownRing: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
  countdownNumber: {
    fontSize: '18px',
    fontWeight: 700,
    color: '#fff',
    textShadow: '0 1px 2px rgba(0,0,0,0.2)',
  },
  processingLoader: {
    display: 'flex',
    alignItems: 'center',
  },
  loaderDots: {
    display: 'flex',
    gap: '6px',
  },
  loaderDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  error: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 16px',
    background: 'rgba(239, 68, 68, 0.15)',
    color: '#ef4444',
    fontSize: '14px',
    borderTop: '1px solid rgba(239, 68, 68, 0.3)',
  },
  dismissButton: {
    background: 'none',
    border: 'none',
    color: '#ef4444',
    fontSize: '20px',
    cursor: 'pointer',
    padding: '0 8px',
    transition: 'opacity 0.2s',
  },
  micContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '32px 24px',
    borderTop: '1px solid rgba(255,255,255,0.1)',
    background: 'rgba(30, 41, 59, 0.5)',
  },
  micButtonWrapper: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  pulseRing: {
    position: 'absolute',
    width: '96px',
    height: '96px',
    borderRadius: '50%',
    border: '2px solid rgba(239, 68, 68, 0.5)',
    pointerEvents: 'none',
  },
  orbitContainer: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
  },
  orbitDot: {
    position: 'absolute',
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: '#fbbf24',
    top: '50%',
    left: '50%',
    marginTop: '-4px',
    marginLeft: '-4px',
  },
  orbitDot2: {
    background: '#f97316',
  },
  orbitDot3: {
    background: '#eab308',
  },
  micButton: {
    position: 'relative',
    width: '96px',
    height: '96px',
    borderRadius: '50%',
    border: 'none',
    background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'transform 0.15s, box-shadow 0.3s',
    boxShadow: '0 4px 20px rgba(59, 130, 246, 0.4)',
    zIndex: 1,
  },
  micButtonRecording: {
    background: 'linear-gradient(135deg, #ef4444, #dc2626)',
    boxShadow: '0 4px 30px rgba(239, 68, 68, 0.5)',
    animation: 'glowPulseRecording 1.5s ease-in-out infinite',
  },
  micButtonProcessing: {
    background: 'linear-gradient(135deg, #f59e0b, #d97706)',
    boxShadow: '0 4px 30px rgba(251, 191, 36, 0.4)',
    animation: 'glowPulseProcessing 2s ease-in-out infinite',
    cursor: 'not-allowed',
  },
  micButtonPlaying: {
    background: 'linear-gradient(135deg, #22c55e, #16a34a)',
    boxShadow: '0 4px 30px rgba(74, 222, 128, 0.4)',
    animation: 'glowPulsePlaying 1s ease-in-out infinite',
    cursor: 'not-allowed',
  },
  micHint: {
    marginTop: '20px',
    fontSize: '14px',
    color: '#94a3b8',
    fontWeight: 500,
  },
};
