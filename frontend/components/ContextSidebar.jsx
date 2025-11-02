import { useState, useEffect } from "react";
import { getGraphStats, searchVectorMemory } from "../utils/api";

export default function ContextSidebar({ context, entities = [] }) {
  const [graphStats, setGraphStats] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [activeTab, setActiveTab] = useState("context");

  useEffect(() => {
    const loadStats = async () => {
      try {
        const stats = await getGraphStats();
        setGraphStats(stats);
      } catch (error) {
        console.error("Failed to load graph stats:", error);
      }
    };
    loadStats();
  }, []);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      const results = await searchVectorMemory(searchQuery, 5);
      if (results.results) {
        setSearchResults(results.results);
        setActiveTab("search");
      }
    } catch (error) {
      console.error("Search failed:", error);
    }
  };

  return (
    <div className="bg-gray-900 p-4 text-white rounded-xl h-full flex flex-col">
      <div className="flex space-x-2 mb-4">
        <button
          onClick={() => setActiveTab("context")}
          className={`px-3 py-1 rounded text-sm ${
            activeTab === "context" ? "bg-blue-600" : "bg-gray-700"
          }`}
        >
          Context
        </button>
        <button
          onClick={() => setActiveTab("search")}
          className={`px-3 py-1 rounded text-sm ${
            activeTab === "search" ? "bg-blue-600" : "bg-gray-700"
          }`}
        >
          Search
        </button>
        <button
          onClick={() => setActiveTab("stats")}
          className={`px-3 py-1 rounded text-sm ${
            activeTab === "stats" ? "bg-blue-600" : "bg-gray-700"
          }`}
        >
          Stats
        </button>
      </div>

      {activeTab === "context" && (
        <div className="flex-1 overflow-y-auto">
          <h2 className="text-xl font-bold mb-2">Memory Context</h2>
          <div className="text-gray-300 text-sm whitespace-pre-wrap">
            {context || "No context yet."}
          </div>
          
          {entities.length > 0 && (
            <div className="mt-4">
              <h3 className="font-bold mb-2">Current Entities:</h3>
              <div className="space-y-1">
                {entities.map((entity, idx) => (
                  <div key={idx} className="bg-gray-800 p-2 rounded text-xs">
                    <div className="font-semibold">{entity.name}</div>
                    {entity.label && (
                      <div className="text-gray-400">Type: {entity.label}</div>
                    )}
                    {entity.relations && entity.relations.length > 0 && (
                      <div className="text-gray-400">
                        Relations: {entity.relations.length}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "search" && (
        <div className="flex-1 overflow-y-auto">
          <h2 className="text-xl font-bold mb-2">Vector Search</h2>
          <div className="flex space-x-2 mb-4">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search memories..."
              className="flex-1 p-2 bg-gray-800 rounded border border-gray-700 text-sm"
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
            <button
              onClick={handleSearch}
              className="bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm"
            >
              Search
            </button>
          </div>
          
          <div className="space-y-2">
            {searchResults.map((result, idx) => (
              <div key={idx} className="bg-gray-800 p-3 rounded">
                <div className="text-sm font-semibold mb-1">
                  Similarity: {(result.similarity * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-300 mb-2">
                  {new Date(result.timestamp).toLocaleString()}
                </div>
                <div className="text-sm">{result.text}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "stats" && (
        <div className="flex-1 overflow-y-auto">
          <h2 className="text-xl font-bold mb-2">Graph Statistics</h2>
          {graphStats ? (
            <div className="space-y-3">
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold">Graph Memory</div>
                <div className="text-sm text-gray-300">
                  Nodes: {graphStats.nodes || 0}
                </div>
                <div className="text-sm text-gray-300">
                  Edges: {graphStats.edges || 0}
                </div>
                <div className="text-sm text-gray-300">
                  Unsaved: {graphStats.unsaved_changes ? "Yes" : "No"}
                </div>
              </div>
              
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold">Vector Memory</div>
                <div className="text-sm text-gray-300">
                  Total Memories: {graphStats.total_memories || 0}
                </div>
                <div className="text-sm text-gray-300">
                  Embedding Dim: {graphStats.embedding_dimension || 0}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-gray-400">Loading stats...</div>
          )}
        </div>
      )}
    </div>
  );
}
