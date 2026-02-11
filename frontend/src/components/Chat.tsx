import React, { useState, useRef } from "react";

interface Message {
    role: "user" | "assistant";
    content: string;
}

export function Chat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState<string>("");
    const [imageBase64, setImageBase64] = useState<string | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = { role: "user", content: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);

        try {
            const response = await fetch("http://127.0.0.1:8000/copilotkit/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: [...messages, userMessage].map((m) => ({
                        role: m.role,
                        content: m.content,
                    })),
                    image_base64: imageBase64 ? imageBase64.split(",")[1] : null, // Send only the base64 data, not the header
                }),
            });
            const data = await response.json();
            const assistantMessage: Message = {
                role: "assistant",
                content: data.response || "Sin respuesta",
            };
            setMessages((prev) => [...prev, assistantMessage]);
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: `❌ Error: ${error}` },
            ]);
        } finally {
            setIsLoading(false);
            setImageBase64(null);
            setImagePreview(null);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            setImageBase64(base64String);
            setImagePreview(base64String);
            setUploadStatus(`✅ Imagen seleccionada: ${file.name}`);
        };
        reader.readAsDataURL(file);
    };

    const handleReindex = async () => {
        setUploadStatus("🔄 Re-indexando...");
        try {
            const response = await fetch("http://127.0.0.1:8000/reindex", {
                method: "POST",
            });
            const data = await response.json();
            setUploadStatus(`✅ ${data.message}`);
        } catch (error) {
            setUploadStatus(`❌ Error: ${error}`);
        }
    };

    return (
        <div style={styles.container}>
            {/* Header */}
            <header style={styles.header}>
                <h1 style={styles.title}>🔬 Histología RAG Multimodal</h1>
                <p style={styles.subtitle}>
                    Asistente inteligente con visión y búsqueda en PDFs
                </p>
            </header>

            <div style={styles.mainContent}>
                {/* Sidebar con controles */}
                <aside style={styles.sidebar}>
                    <div style={styles.card}>
                        <h2 style={styles.cardTitle}>📂 Gestión de Archivos</h2>

                        <div style={styles.buttonGroup}>
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                style={styles.uploadButton}
                            >
                                🖼️ Seleccionar Imagen
                            </button>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                onChange={handleFileUpload}
                                style={{ display: "none" }}
                            />

                            <button onClick={handleReindex} style={styles.reindexButton}>
                                🔄 Re-indexar PDFs
                            </button>
                        </div>

                        {uploadStatus && (
                            <div style={styles.statusBox}>{uploadStatus}</div>
                        )}
                    </div>

                    <div style={styles.card}>
                        <h2 style={styles.cardTitle}>💡 Instrucciones</h2>
                        <ul style={styles.instructionList}>
                            <li>Sube una imagen histológica para análisis</li>
                            <li>Haz preguntas sobre tus documentos PDF</li>
                            <li>Añade PDFs a <code>./pdfs</code> y re-indexa</li>
                        </ul>
                        <h2 style={styles.cardTitle}>💡 Instrucciones</h2>
                        <ul style={styles.instructionList}>
                            <li>Sube una imagen histológica para análisis</li>
                            <li>Haz preguntas sobre tus documentos PDF</li>
                            <li>Añade PDFs a <code>./pdfs</code> y re-indexa</li>
                        </ul>

                        {imagePreview && (
                            <div style={{ marginTop: '20px', textAlign: 'center' }}>
                                <p style={{ fontSize: '12px', marginBottom: '5px' }}>Vista previa:</p>
                                <img src={imagePreview} alt="Preview" style={{ maxWidth: '100%', borderRadius: '8px', maxHeight: '150px' }} />
                            </div>
                        )}
                    </div>
                </aside>

                {/* Chat principal */}
                <main style={styles.chatContainer}>
                    <div style={styles.messagesContainer}>
                        {messages.length === 0 ? (
                            <div style={styles.emptyState}>
                                <span style={styles.emptyIcon}>💬</span>
                                <p>Hola, soy tu asistente de histología.</p>
                                <p style={styles.emptyHint}>
                                    ¿En qué puedo ayudarte hoy?
                                </p>
                            </div>
                        ) : (
                            messages.map((msg, idx) => (
                                <div
                                    key={idx}
                                    style={{
                                        ...styles.message,
                                        ...(msg.role === "user"
                                            ? styles.userMessage
                                            : styles.assistantMessage),
                                    }}
                                >
                                    <span style={styles.messageRole}>
                                        {msg.role === "user" ? "👤 Tú" : "🤖 Asistente"}
                                    </span>
                                    <p style={styles.messageContent}>{msg.content}</p>
                                </div>
                            ))
                        )}
                        {isLoading && (
                            <div style={{ ...styles.message, ...styles.assistantMessage }}>
                                <span style={styles.messageRole}>🤖 Asistente</span>
                                <p style={styles.messageContent}>⏳ Pensando...</p>
                            </div>
                        )}
                    </div>

                    <div style={styles.inputContainer}>
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Escribe tu pregunta..."
                            style={styles.input}
                            disabled={isLoading}
                        />
                        <button
                            onClick={handleSend}
                            disabled={isLoading || !input.trim()}
                            style={{
                                ...styles.sendButton,
                                opacity: isLoading || !input.trim() ? 0.5 : 1,
                            }}
                        >
                            Enviar
                        </button>
                    </div>
                </main>
            </div>
        </div>
    );
}

const styles: { [key: string]: React.CSSProperties } = {
    container: {
        minHeight: "100vh",
        background: "linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%)",
        color: "#1b5e20",
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    },
    header: {
        padding: "24px 32px",
        borderBottom: "1px solid rgba(27, 94, 32, 0.15)",
        background: "rgba(255, 255, 255, 0.7)",
        backdropFilter: "blur(10px)",
    },
    title: {
        margin: 0,
        fontSize: "28px",
        fontWeight: 700,
        background: "linear-gradient(90deg, #2e7d32 0%, #43a047 50%, #66bb6a 100%)",
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
    },
    subtitle: {
        margin: "8px 0 0",
        color: "#4caf50",
        fontSize: "14px",
    },
    mainContent: {
        display: "flex",
        height: "calc(100vh - 100px)",
        gap: "24px",
        padding: "24px",
    },
    sidebar: {
        width: "300px",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        flexShrink: 0,
    },
    card: {
        background: "rgba(255, 255, 255, 0.8)",
        borderRadius: "16px",
        padding: "20px",
        border: "1px solid rgba(76, 175, 80, 0.3)",
        boxShadow: "0 4px 20px rgba(76, 175, 80, 0.15)",
        backdropFilter: "blur(10px)",
    },
    cardTitle: {
        margin: "0 0 16px",
        fontSize: "16px",
        fontWeight: 600,
        color: "#2e7d32",
    },
    buttonGroup: {
        display: "flex",
        flexDirection: "column",
        gap: "12px",
    },
    uploadButton: {
        padding: "12px 16px",
        borderRadius: "10px",
        border: "none",
        background: "linear-gradient(135deg, #43a047 0%, #66bb6a 100%)",
        color: "#fff",
        fontWeight: 600,
        cursor: "pointer",
        fontSize: "14px",
        transition: "transform 0.2s, box-shadow 0.2s",
        boxShadow: "0 4px 15px rgba(67, 160, 71, 0.4)",
    },
    reindexButton: {
        padding: "12px 16px",
        borderRadius: "10px",
        border: "1px solid rgba(76, 175, 80, 0.4)",
        background: "rgba(255, 255, 255, 0.6)",
        color: "#2e7d32",
        fontWeight: 600,
        cursor: "pointer",
        fontSize: "14px",
        transition: "background 0.2s",
    },
    statusBox: {
        marginTop: "12px",
        padding: "12px",
        borderRadius: "8px",
        background: "rgba(200, 230, 201, 0.6)",
        fontSize: "13px",
        color: "#1b5e20",
        border: "1px solid rgba(76, 175, 80, 0.3)",
    },
    instructionList: {
        margin: 0,
        paddingLeft: "20px",
        fontSize: "13px",
        color: "#388e3c",
        lineHeight: 1.8,
    },
    chatContainer: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        background: "rgba(255, 255, 255, 0.7)",
        borderRadius: "16px",
        border: "1px solid rgba(76, 175, 80, 0.3)",
        overflow: "hidden",
        boxShadow: "0 4px 20px rgba(76, 175, 80, 0.1)",
        backdropFilter: "blur(10px)",
    },
    messagesContainer: {
        flex: 1,
        overflowY: "auto",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
    },
    emptyState: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        color: "#4caf50",
        textAlign: "center",
    },
    emptyIcon: {
        fontSize: "48px",
        marginBottom: "16px",
    },
    emptyHint: {
        fontSize: "14px",
        color: "#81c784",
    },
    message: {
        padding: "16px",
        borderRadius: "12px",
        maxWidth: "80%",
    },
    userMessage: {
        alignSelf: "flex-end",
        background: "linear-gradient(135deg, #43a047 0%, #66bb6a 100%)",
        color: "#fff",
    },
    assistantMessage: {
        alignSelf: "flex-start",
        background: "rgba(200, 230, 201, 0.8)",
        color: "#1b5e20",
        border: "1px solid rgba(76, 175, 80, 0.3)",
    },
    messageRole: {
        display: "block",
        fontSize: "12px",
        opacity: 0.7,
        marginBottom: "6px",
    },
    messageContent: {
        margin: 0,
        lineHeight: 1.6,
        whiteSpace: "pre-wrap",
    },
    inputContainer: {
        display: "flex",
        gap: "12px",
        padding: "16px 24px",
        borderTop: "1px solid rgba(76, 175, 80, 0.2)",
        background: "rgba(255, 255, 255, 0.8)",
    },
    input: {
        flex: 1,
        padding: "14px 18px",
        borderRadius: "12px",
        border: "1px solid rgba(76, 175, 80, 0.3)",
        background: "rgba(255, 255, 255, 0.9)",
        color: "#1b5e20",
        fontSize: "14px",
        outline: "none",
    },
    sendButton: {
        padding: "14px 28px",
        borderRadius: "12px",
        border: "none",
        background: "linear-gradient(135deg, #43a047 0%, #66bb6a 100%)",
        color: "#fff",
        fontWeight: 600,
        cursor: "pointer",
        fontSize: "14px",
        boxShadow: "0 4px 15px rgba(67, 160, 71, 0.4)",
    },
};
