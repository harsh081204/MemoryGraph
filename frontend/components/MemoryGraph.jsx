"use client";
import ReactFlow, { 
  MiniMap, 
  Controls, 
  Background, 
  useNodesState, 
  useEdgesState,
  MarkerType
} from "reactflow";
import "reactflow/dist/style.css";
import { useState, useEffect, useCallback } from "react";
import { getCentralEntities, getRelatedEntities } from "../utils/api";

export default function MemoryGraph({ entities }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [centralEntities, setCentralEntities] = useState([]);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [relatedEntities, setRelatedEntities] = useState([]);

  // Load central entities on component mount
  useEffect(() => {
    const loadCentralEntities = async () => {
      try {
        const data = await getCentralEntities();
        if (data.central_entities) {
          setCentralEntities(data.central_entities);
        }
      } catch (error) {
        console.error("Failed to load central entities:", error);
      }
    };
    loadCentralEntities();
  }, []);

  // Update graph when entities change
  useEffect(() => {
    if (entities && entities.length > 0) {
      const newNodes = entities.map((ent, idx) => ({
        id: ent.name,
        data: { 
          label: ent.name,
          type: ent.label || 'ENTITY',
          centrality: centralEntities.find(ce => ce.entity === ent.name)?.centrality_score || 0
        },
        position: { 
          x: Math.cos(idx * 2 * Math.PI / entities.length) * 200 + 300, 
          y: Math.sin(idx * 2 * Math.PI / entities.length) * 200 + 200 
        },
        style: {
          background: centralEntities.find(ce => ce.entity === ent.name) ? '#3b82f6' : '#6b7280',
          color: 'white',
          border: '2px solid #1f2937',
          borderRadius: '8px',
          fontSize: '12px',
          fontWeight: 'bold'
        }
      }));

      const newEdges = [];
      entities.forEach((ent) => {
        ent.relations?.forEach((rel, i) => {
          newEdges.push({
            id: `${ent.name}-${rel.target}-${i}`,
            source: ent.name,
            target: rel.target,
            label: rel.type,
            animated: true,
            style: { stroke: '#3b82f6', strokeWidth: 2 },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: '#3b82f6',
            },
          });
        });
      });

      setNodes(newNodes);
      setEdges(newEdges);
    }
  }, [entities, centralEntities]);

  const handleNodeClick = useCallback(async (event, node) => {
    setSelectedEntity(node.data.label);
    try {
      const data = await getRelatedEntities(node.data.label);
      if (data.related_entities) {
        setRelatedEntities(data.related_entities);
      }
    } catch (error) {
      console.error("Failed to load related entities:", error);
    }
  }, []);

  const nodeTypes = {
    central: {
      style: {
        background: '#10b981',
        color: 'white',
        border: '3px solid #059669',
        borderRadius: '12px',
        fontSize: '14px',
        fontWeight: 'bold'
      }
    }
  };

  return (
    <div className="w-full h-[500px] bg-gray-800 rounded-xl shadow-md relative">
      <div className="absolute top-2 left-2 z-10 bg-gray-900 text-white p-2 rounded-lg text-xs">
        <div>Nodes: {nodes.length}</div>
        <div>Edges: {edges.length}</div>
        {selectedEntity && (
          <div className="mt-2">
            <div className="font-bold">Selected: {selectedEntity}</div>
            <div>Related: {relatedEntities.length} entities</div>
          </div>
        )}
      </div>
      
      <ReactFlow 
        nodes={nodes} 
        edges={edges} 
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
      >
        <MiniMap 
          nodeColor={(node) => node.style?.background || '#6b7280'}
          style={{ background: '#1f2937' }}
        />
        <Controls />
        <Background color="#374151" gap={20} />
      </ReactFlow>
    </div>
  );
}
