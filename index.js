import express from "express";
import cors from "cors";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Configuración de Azure OpenAI
const endpoint = process.env.AZURE_OPENAI_ENDPOINT;
const deployment_name = process.env.AZURE_DEPLOYMENT_NAME;
const api_key = process.env.AZURE_OPENAI_KEY;

// Configuración de Azure AI Search
const searchEndpoint = process.env.AZURE_SEARCH_ENDPOINT;
const searchKey = process.env.AZURE_SEARCH_KEY;
const searchIndex = process.env.AZURE_SEARCH_INDEX;

// Configuración de Embeddings
const embeddingEndpoint = process.env.AZURE_EMBEDDING_ENDPOINT;
const embeddingKey = process.env.AZURE_EMBEDDING_KEY;
const embeddingDeployment = process.env.AZURE_EMBEDDING_DEPLOYMENT;

const client = new OpenAI({
    baseURL: endpoint,
    apiKey: api_key,
    defaultHeaders: {
      "api-key": api_key,
    }
});

app.get("/", (req, res) => {
  res.json({ 
    status: "ok", 
    message: "Azure OpenAI API Backend con datos personalizados",
    config: {
      searchIndex: searchIndex,
      embeddingModel: embeddingDeployment,
      chatModel: deployment_name
    }
  });
});

// Endpoint para chat CON tus datos de Azure AI Search
app.post("/api/chat", async (req, res) => {
  try {
    const { messages, systemPrompt, queryType = "vector_simple_hybrid" } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ 
        error: "Se requiere un array de mensajes" 
      });
    }

    // Construir mensajes
    const fullMessages = systemPrompt 
      ? [{ role: "system", content: systemPrompt }, ...messages]
      : messages;

    // Configuración con Azure AI Search + Embeddings
    const requestBody = {
      messages: fullMessages,
      temperature: 0.7,
      max_tokens: 1000,
      data_sources: [
        {
          type: "azure_search",
          parameters: {
            endpoint: searchEndpoint,
            index_name: searchIndex,
            authentication: {
              type: "api_key",
              key: searchKey
            },
            // CONFIGURACIÓN DE EMBEDDINGS (ESTO ES LO QUE FALTABA)
            embedding_dependency: {
              type: "deployment_name",
              deployment_name: embeddingDeployment
            },
            query_type: queryType, // "vector_simple_hybrid", "simple", "semantic", "vector"
            in_scope: true,
            role_information: systemPrompt || "Eres un asistente útil que responde basándose en los datos proporcionados.",
            strictness: 3, // 1-5
            top_n_documents: 5
          }
        }
      ]
    };

    // Construir la URL correcta para el endpoint
    const baseUrl = endpoint.replace('/openai/v1', '');
    const apiUrl = `${baseUrl}/openai/deployments/${deployment_name}/chat/completions?api-version=2024-02-15-preview`;

    console.log('🔍 Haciendo petición a:', apiUrl);
    console.log('📦 Request body:', JSON.stringify(requestBody, null, 2));

    // Hacer la petición
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'api-key': api_key
      },
      body: JSON.stringify(requestBody)
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('❌ Error response:', data);
      throw new Error(data.error?.message || JSON.stringify(data));
    }

    // Extraer la respuesta y las referencias
    const message = data.choices[0].message;
    const citations = message.context?.citations || [];

    console.log('✅ Respuesta exitosa con', citations.length, 'referencias');

    res.json({
      success: true,
      message: message.content,
      citations: citations,
      usage: data.usage
    });

  } catch (error) {
    console.error("❌ Error en /api/chat:", error);
    res.status(500).json({ 
      success: false,
      error: "Error al procesar la solicitud",
      details: error.message 
    });
  }
});

// Endpoint SIN usar tus datos (modelo base)
app.post("/api/chat/base", async (req, res) => {
  try {
    const { messages, systemPrompt = "Eres un asistente útil." } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ 
        error: "Se requiere un array de mensajes" 
      });
    }

    const fullMessages = [
      { role: "system", content: systemPrompt },
      ...messages
    ];

    const completion = await client.chat.completions.create({
      messages: fullMessages,
      model: deployment_name,
      temperature: 0.7,
      max_tokens: 1000
    });

    res.json({
      success: true,
      message: completion.choices[0].message.content,
      usage: completion.usage
    });

  } catch (error) {
    console.error("Error en /api/chat/base:", error);
    res.status(500).json({ 
      success: false,
      error: "Error al procesar la solicitud",
      details: error.message 
    });
  }
});

// Endpoint para probar solo embeddings
app.post("/api/embeddings", async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: "Se requiere un texto" });
    }

    const response = await fetch(`${embeddingEndpoint}/openai/deployments/${embeddingDeployment}/embeddings?api-version=2023-05-15`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'api-key': embeddingKey
      },
      body: JSON.stringify({
        input: text
      })
    });

    const data = await response.json();

    res.json({
      success: true,
      embedding: data.data[0].embedding,
      dimensions: data.data[0].embedding.length
    });

  } catch (error) {
    console.error("Error en /api/embeddings:", error);
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

app.use((req, res) => {
  res.status(404).json({ error: "Ruta no encontrada" });
});

app.listen(PORT, () => {
  console.log(`✅ Servidor ejecutándose en http://localhost:${PORT}`);
  console.log(`📡 Endpoint con tus datos: POST http://localhost:${PORT}/api/chat`);
  console.log(`📡 Endpoint modelo base: POST http://localhost:${PORT}/api/chat/base`);
  console.log(`📡 Endpoint embeddings: POST http://localhost:${PORT}/api/embeddings`);
  console.log(`\n🔧 Configuración:`);
  console.log(`   - Chat Model: ${deployment_name}`);
  console.log(`   - Embedding Model: ${embeddingDeployment}`);
  console.log(`   - Search Index: ${searchIndex}`);
});