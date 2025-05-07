// src/pages/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet';
import Link from 'next/link';
import useExchangeData from '../hooks/useExchangeData';
import CountyDistributionMap from '../components/vizualizari/CountyDistributionMap';
import BucharestSectorsMap from '../components/vizualizari/BucharestSectorsMap';
import FinancialAnalysis from '../components/vizualizari/FinancialAnalysis';
import DataAnalysisLLM from '../components/analiza/DataAnalysisLLM';
import FilterPanel from '../components/filtre/FilterPanel';
import StatusChart from '../components/vizualizari/StatusChart';
import TopExchangeHouses from '../components/vizualizari/TopExchangeHouses';
import TimelineChart from '../components/vizualizari/TimelineChart';
import PrivateRoute from '../components/auth/PrivateRoute';
import { useAuth } from '../context/AuthContext';

const Dashboard: React.FC = () => {
  // Auth context pentru informații utilizator
  const { currentUser } = useAuth();
  
  // Hook pentru datele aplicației
  const {
    caseSchimb,
    puncteSchimb,
    dateFinanciare,
    isLoading,
    hasError,
    errors,
    refreshAllData,
    isPollingEnabled,
    togglePolling,
    filters,
    updateFilters,
    financialMetrics
  } = useExchangeData();
  
  // State pentru tab-uri
  const [activeTab, setActiveTab] = useState<'general' | 'geo' | 'financiar' | 'analiza-ai'>('general');
  const [geoTab, setGeoTab] = useState<'judete' | 'bucuresti'>('judete');
  const [finTab, setFinTab] = useState<'profit' | 'contributii'>('profit');
  
  // State pentru ultima actualizare
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  
  // Actualizare timestamp la încărcarea datelor
  useEffect(() => {
    if (!isLoading) {
      setLastUpdate(new Date());
    }
  }, [isLoading, caseSchimb, puncteSchimb, dateFinanciare]);
  
  // Funcție pentru actualizarea manuală a datelor
  const handleRefresh = () => {
    refreshAllData();
  };
  
  // Calculare statistici pentru panoul general
  const totalCaseActive = caseSchimb.filter(casa => !casa["DATA INCHIDERE CASA"]).length;
  const totalPuncteActive = puncteSchimb.filter(punct => !punct["DATA INCHIDERE PUNCT"]).length;
  const uniqueJudete = new Set(puncteSchimb.map(punct => punct["JUDET / SECTOR PUNCT"])).size;
  
  if (isLoading && !caseSchimb.length) {
    return (
      <PrivateRoute>
        <div className="flex items-center justify-center h-screen">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-lg">Se încarcă datele, vă rugăm așteptați...</p>
          </div>
        </div>
      </PrivateRoute>
    );
  }
  
  return (
    <PrivateRoute>
      <div className="dashboard-container bg-gray-50 min-h-screen pb-8">
        <Helmet>
          <title>Dashboard | Case de Schimb Valutar</title>
        </Helmet>
        
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-900">
              Dashboard Case de Schimb Valutar
            </h1>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">
                Ultima actualizare: {lastUpdate.toLocaleString()}
              </span>
              <button
                onClick={handleRefresh}
                disabled={isLoading}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 text-sm"
              >
                {isLoading ? 'Se încarcă...' : 'Actualizare'}
              </button>
              <div className="flex items-center">
                <span className="mr-2 text-sm">Auto-refresh:</span>
                <button
                  onClick={togglePolling}
                  className={`w-10 h-5 rounded-full p-1 transition-colors duration-300 ease-in-out ${
                    isPollingEnabled ? 'bg-green-500' : 'bg-gray-300'
                  }`}
                >
                  <div
                    className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform duration-300 ease-in-out ${
                      isPollingEnabled ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  ></div>
                </button>
              </div>
              <div className="relative">
                <Link 
                  href="/profile"
                  className="inline-flex items-center px-3 py-1 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  {currentUser?.email?.split('@')[0]}
                </Link>
              </div>
            </div>
          </div>
        </header>
        
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
          {hasError && (
            <div className="mb-6 bg-red-50 p-4 rounded-lg border border-red-300 text-red-700">
              <h3 className="text-lg font-medium">Eroare la încărcarea datelor</h3>
              <p>{errors.map(e => e?.toString()).join(', ')}</p>
              <button
                onClick={handleRefresh}
                className="mt-2 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Încercați din nou
              </button>
            </div>
          )}
          
          {/* Carduri cu sumar statistici */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow p-4 border-l-4 border-blue-500">
              <div className="flex justify-between">
                <div>
                  <p className="text-sm text-gray-500">Case de Schimb</p>
                  <p className="text-2xl font-bold">{caseSchimb.length}</p>
                </div>
                <div className="rounded-full p-2 bg-blue-100 h-10 w-10 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                </div>
              </div>
              <p className="mt-2 text-sm text-green-600">{totalCaseActive} active</p>
            </div>
            
            <div className="bg-white rounded-lg shadow p-4 border-l-4 border-green-500">
              <div className="flex justify-between">
                <div>
                  <p className="text-sm text-gray-500">Puncte de Schimb</p>
                  <p className="text-2xl font-bold">{puncteSchimb.length}</p>
                </div>
                <div className="rounded-full p-2 bg-green-100 h-10 w-10 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
              </div>
              <p className="mt-2 text-sm text-green-600">{totalPuncteActive} active</p>
            </div>
            
            <div className="bg-white rounded-lg shadow p-4 border-l-4 border-purple-500">
              <div className="flex justify-between">
                <div>
                  <p className="text-sm text-gray-500">Acoperire Teritorială</p>
                  <p className="text-2xl font-bold">{uniqueJudete}</p>
                </div>
                <div className="rounded-full p-2 bg-purple-100 h-10 w-10 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                </div>
              </div>
              <p className="mt-2 text-sm text-purple-600">județe/sectoare</p>
            </div>
            
            <div className="bg-white rounded-lg shadow p-4 border-l-4 border-amber-500">
              <div className="flex justify-between">
                <div>
                  <p className="text-sm text-gray-500">Profit Total</p>
                  <p className="text-2xl font-bold">
                    {financialMetrics ? 
                      financialMetrics.profitTotal.toLocaleString() + ' RON' : 
                      'Indisponibil'}
                  </p>
                </div>
                <div className="rounded-full p-2 bg-amber-100 h-10 w-10 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
              <p className="mt-2 text-sm text-amber-600">
                {financialMetrics ? 
                  `${financialMetrics.numarCaseProfitabile} case profitabile` : 
                  'Indisponibil'}
              </p>
            </div>
          </div>
          
          {/* Panoul de filtrare */}
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Filtre</h3>
              <FilterPanel filters={filters} updateFilters={updateFilters} />
            </div>
          </div>
          
          {/* Tab-uri principale */}
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="border-b border-gray-200">
              <nav className="flex -mb-px">
                <button
                  className={`py-4 px-6 text-center font-medium ${
                    activeTab === 'general'
                      ? 'border-b-2 border-blue-500 text-blue-600'
                      : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => setActiveTab('general')}
                >
                  Statistici Generale
                </button>
                
                <button
                  className={`py-4 px-6 text-center font-medium ${
                    activeTab === 'geo'
                      ? 'border-b-2 border-blue-500 text-blue-600'
                      : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => setActiveTab('geo')}
                >
                  Distribuție Geografică
                </button>
                
                <button
                  className={`py-4 px-6 text-center font-medium ${
                    activeTab === 'financiar'
                      ? 'border-b-2 border-blue-500 text-blue-600'
                      : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => setActiveTab('financiar')}
                >
                  Analiză Financiară
                </button>
                
                <button
                  className={`py-4 px-6 text-center font-medium ${
                    activeTab === 'analiza-ai'
                      ? 'border-b-2 border-blue-500 text-blue-600'
                      : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => setActiveTab('analiza-ai')}
                >
                  Analiză AI
                </button>
              </nav>
            </div>
            
            {/* Conținutul tab-ului activ */}
            <div className="p-4 md:p-6">
              {activeTab === 'general' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-white p-4 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Tipuri de puncte de schimb</h3>
                    <StatusChart />
                  </div>
                  
                  <div className="bg-white p-4 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Top case după număr de puncte</h3>
                    <TopExchangeHouses />
                  </div>
                  
                  <div className="bg-white p-4 rounded-lg border md:col-span-2">
                    <h3 className="text-lg font-semibold mb-4">Evoluție temporală</h3>
                    <TimelineChart />
                  </div>
                </div>
              )}
              
              {activeTab === 'geo' && (
                <div>
                  <div className="mb-4 flex space-x-2">
                    <button
                      className={`px-4 py-2 rounded ${
                        geoTab === 'judete' ? 'bg-blue-500 text-white' : 'bg-gray-200'
                      }`}
                      onClick={() => setGeoTab('judete')}
                    >
                      Județe
                    </button>
                    <button
                      className={`px-4 py-2 rounded ${
                        geoTab === 'bucuresti' ? 'bg-blue-500 text-white' : 'bg-gray-200'
                      }`}
                      onClick={() => setGeoTab('bucuresti')}
                    >
                      București
                    </button>
                  </div>
                  
                  {geoTab === 'judete' && <CountyDistributionMap />}
                  {geoTab === 'bucuresti' && <BucharestSectorsMap />}
                </div>
              )}
              
              {activeTab === 'financiar' && (
                <FinancialAnalysis />
              )}
              
              {activeTab === 'analiza-ai' && (
                <DataAnalysisLLM />
              )}
            </div>
          </div>
        </main>
      </div>
    </PrivateRoute>
  );
};

export default Dashboard;