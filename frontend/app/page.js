"use client";

import { useState } from "react";
import ChatBox from "../components/ChatBox";
import MemoryGraph from "../components/MemoryGraph";
import ContextSidebar from "../components/ContextSidebar";

export default function Home() {
  const [entities, setEntities] = useState([]);
  const [context, setContext] = useState("");

  const handleUpdateGraph = (newEntities, newContext) => {
    setEntities(newEntities);
    setContext(newContext);
  };

  return (
    <div className="grid grid-cols-12 gap-4 p-6 bg-gray-950 min-h-screen">
      <div className="col-span-4">
        <ChatBox onUpdateGraph={handleUpdateGraph} />
      </div>
      <div className="col-span-5">
        <MemoryGraph entities={entities} />
      </div>
      <div className="col-span-3">
        <ContextSidebar context={context} entities={entities} />
      </div>
    </div>
  );
}
