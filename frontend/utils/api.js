export async function sendMessage(message) {
  try {
    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error("Failed to connect to backend");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("API Error:", error);
    return { error: error.message };
  }
}

export async function getCentralEntities() {
  try {
    const response = await fetch("http://127.0.0.1:8000/graph/central-entities");
    if (!response.ok) throw new Error("Failed to fetch central entities");
    return await response.json();
  } catch (error) {
    console.error("Central entities error:", error);
    return { error: error.message };
  }
}

export async function getRelatedEntities(entityName, maxDistance = 2) {
  try {
    const response = await fetch(`http://127.0.0.1:8000/graph/related/${encodeURIComponent(entityName)}?max_distance=${maxDistance}`);
    if (!response.ok) throw new Error("Failed to fetch related entities");
    return await response.json();
  } catch (error) {
    console.error("Related entities error:", error);
    return { error: error.message };
  }
}

export async function searchVectorMemory(query, topK = 5) {
  try {
    const response = await fetch("http://127.0.0.1:8000/vector/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!response.ok) throw new Error("Failed to search vector memory");
    return await response.json();
  } catch (error) {
    console.error("Vector search error:", error);
    return { error: error.message };
  }
}

export async function getGraphStats() {
  try {
    const response = await fetch("http://127.0.0.1:8000/graph/stats");
    if (!response.ok) throw new Error("Failed to fetch graph stats");
    return await response.json();
  } catch (error) {
    console.error("Graph stats error:", error);
    return { error: error.message };
  }
}