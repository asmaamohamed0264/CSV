import React from 'react';

const ArchitectureDiagram = () => {
  return (
    <div className="bg-gray-100 p-6 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-center text-blue-800">Arhitectura Aplicației - Case de Schimb Valutar</h2>
      
      {/* Main Container */}
      <div className="flex flex-col space-y-4">
        
        {/* Frontend Layer */}
        <div className="border-2 border-blue-500 rounded-lg p-4 bg-blue-50">
          <h3 className="text-xl font-semibold mb-3 text-blue-700">Frontend (React + TypeScript)</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border border-blue-300 rounded p-3 bg-white">
              <h4 className="font-medium text-blue-600 mb-2">Pagini</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>Dashboard Principal</li>
                <li>Harta Distribuție</li>
                <li>Analiză Financiară</li>
                <li>Rapoarte</li>
                <li>Administrare</li>
              </ul>
            </div>
            <div className="border border-blue-300 rounded p-3 bg-white">
              <h4 className="font-medium text-blue-600 mb-2">Componente Reutilizabile</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>Grafice (Recharts/D3)</li>
                <li>Filtre</li>
                <li>Tabele Dinamice</li>
                <li>Hărți Interactive</li>
                <li>Panouri de Control</li>
              </ul>
            </div>
            <div className="border border-blue-300 rounded p-3 bg-white">
              <h4 className="font-medium text-blue-600 mb-2">State Management</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>Context API</li>
                <li>React Query</li>
                <li>Custom Hooks</li>
                <li>Reducere (optional)</li>
              </ul>
            </div>
          </div>
        </div>
        
        {/* Data Layer */}
        <div className="border-2 border-green-500 rounded-lg p-4 bg-green-50">
          <h3 className="text-xl font-semibold mb-3 text-green-700">Gestionare Date</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border border-green-300 rounded p-3 bg-white">
              <h4 className="font-medium text-green-600 mb-2">Servicii Date</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>API Google Sheets</li>
                <li>Custom Hooks pentru Fetch</li>
                <li>Normalizare Date</li>
                <li>Validare Date</li>
              </ul>
            </div>
            <div className="border border-green-300 rounded p-3 bg-white">
              <h4 className="font-medium text-green-600 mb-2">Strategii Optimizare</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>Caching</li>
                <li>Sincronizare Selectivă</li>
                <li>Polling Inteligent</li>
                <li>WebSockets (opțional)</li>
                <li>Versionare Date</li>
              </ul>
            </div>
            <div className="border border-green-300 rounded p-3 bg-white">
              <h4 className="font-medium text-green-600 mb-2">Utilități</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>Normalizare Diacritice</li>
                <li>Fuzzy Matching</li>
                <li>Agregare Date</li>
                <li>Transformări</li>
              </ul>
            </div>
          </div>
        </div>
        
        {/* Integration Layer */}
        <div className="border-2 border-purple-500 rounded-lg p-4 bg-purple-50">
          <h3 className="text-xl font-semibold mb-3 text-purple-700">Integrări</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border border-purple-300 rounded p-3 bg-white">
              <h4 className="font-medium text-purple-600 mb-2">Autentificare</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>Firebase Auth</li>
                <li>Auth0 (alternativă)</li>
                <li>Role-based Access</li>
              </ul>
            </div>
            <div className="border border-purple-300 rounded p-3 bg-white">
              <h4 className="font-medium text-purple-600 mb-2">LLM APIs</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>Middleware pentru Chei API</li>
                <li>Router LLM Unificat</li>
                <li>Generare Rapoarte</li>
                <li>Streaming Răspunsuri</li>
              </ul>
            </div>
            <div className="border border-purple-300 rounded p-3 bg-white">
              <h4 className="font-medium text-purple-600 mb-2">Deployment</h4>
              <ul className="list-disc pl-4 text-sm">
                <li>Vercel</li>
                <li>CI/CD via GitHub</li>
                <li>Variabile de Mediu</li>
                <li>Feature Flags</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      
      {/* Flow arrows */}
      <div className="mt-4 flex justify-center">
        <svg width="100%" height="60" viewBox="0 0 400 60" className="max-w-md">
          <defs>
            <marker 
              id="arrowhead" 
              markerWidth="10" 
              markerHeight="7" 
              refX="0" 
              refY="3.5" 
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#4B5563" />
            </marker>
          </defs>
          <path 
            d="M 50,10 L 350,10 M 350,10 L 350,50 M 350,50 L 50,50 M 50,50 L 50,10" 
            stroke="#4B5563" 
            strokeWidth="2" 
            fill="none" 
            markerEnd="url(#arrowhead)" 
            strokeDasharray="5,5"
          />
        </svg>
      </div>
      
      {/* Legend */}
      <div className="mt-6 border rounded-lg p-4 bg-white">
        <h3 className="text-lg font-semibold mb-2">Fluxul Datelor</h3>
        <div className="flex flex-col space-y-2 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
            <span>Frontend Components - Afișare date și interacțiuni utilizator</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-500 rounded-full mr-2"></div>
            <span>Data Layer - Comunicare cu sursele de date și normalizare</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-purple-500 rounded-full mr-2"></div>
            <span>Integrări - Autentificare, LLM și deployment</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ArchitectureDiagram;
