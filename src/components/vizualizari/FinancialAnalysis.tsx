// src/components/vizualizari/FinancialAnalysis.tsx
import React, { useMemo, useState } from 'react';
import { useExchangeData } from '../../hooks/useExchangeData';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, PieChart, Pie, Cell, ScatterChart, Scatter
} from 'recharts';
import FinancialDataTable from './FinancialDataTable';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const FinancialAnalysis: React.FC = () => {
  const { dateFinanciare, financialMetrics, puncteSchimb } = useExchangeData();
  const [activeView, setActiveView] = useState<'profit' | 'contributors' | 'correlation' | 'data-table'>('profit');
  
  // Obține anul de referință cel mai recent
  const latestYear = useMemo(() => {
    if (!dateFinanciare.length) return null;
    return Math.max(...dateFinanciare.map(item => item["AN DE REFERINTA"]));
  }, [dateFinanciare]);
  
  // Pregătirea datelor pentru graficul de profitabilitate
  const profitabilityData = useMemo(() => {
    if (!dateFinanciare.length || !latestYear) return [];
    
    return dateFinanciare
      .filter(item => 
        item["AN DE REFERINTA"] === latestYear && 
        item["ARE_BILANT_DA/NU"] === "DA" && 
        item["PROFIT_PIERDERE"] !== ""
      )
      .map(item => ({
        name: item["DENUMIRE"].length > 20 
          ? item["DENUMIRE"].substring(0, 17) + '...'
          : item["DENUMIRE"],
        fullName: item["DENUMIRE"],
        profit: typeof item["PROFIT_PIERDERE"] === 'string'
          ? parseFloat(item["PROFIT_PIERDERE"])
          : item["PROFIT_PIERDERE"]
      }))
      .sort((a, b) => b.profit - a.profit)
      .slice(0, 10); // Top 10 după profit
  }, [dateFinanciare, latestYear]);
  
  // Pregătirea datelor pentru graficul de contribuții la buget
  const budgetContributionData = useMemo(() => {
    if (!dateFinanciare.length || !latestYear) return [];
    
    return dateFinanciare
      .filter(item => 
        item["AN DE REFERINTA"] === latestYear && 
        item["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] !== ""
      )
      .map(item => ({
        name: item["DENUMIRE"].length > 20 
          ? item["DENUMIRE"].substring(0, 17) + '...'
          : item["DENUMIRE"],
        fullName: item["DENUMIRE"],
        contributie: typeof item["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] === 'string'
          ? parseFloat(item["INCASARI LA BUGETUL GENESAL CONSOLIDAT"])
          : item["INCASARI LA BUGETUL GENESAL CONSOLIDAT"]
      }))
      .sort((a, b) => b.contributie - a.contributie)
      .slice(0, 10); // Top 10 după contribuții
  }, [dateFinanciare, latestYear]);
  
  // Pregătirea datelor pentru corelația angajați-profit
  const correlationData = useMemo(() => {
    if (!dateFinanciare.length || !latestYear) return [];
    
    return dateFinanciare
      .filter(item => 
        item["AN DE REFERINTA"] === latestYear && 
        item["ARE_BILANT_DA/NU"] === "DA" &&
        item["PROFIT_PIERDERE"] !== "" &&
        item["NR_MEDIU SALARIATI"] !== ""
      )
      .map(item => ({
        name: item["DENUMIRE"],
        angajati: typeof item["NR_MEDIU SALARIATI"] === 'string'
          ? parseInt(item["NR_MEDIU SALARIATI"], 10)
          : item["NR_MEDIU SALARIATI"],
        profit: typeof item["PROFIT_PIERDERE"] === 'string'
          ? parseFloat(item["PROFIT_PIERDERE"])
          : item["PROFIT_PIERDERE"],
        fondSalarii: typeof item["FOND_SALARII"] === 'string'
          ? parseFloat(item["FOND_SALARII"])
          : item["FOND_SALARII"]
      }));
  }, [dateFinanciare, latestYear]);
  
  // Date pentru distribuția profitabilă vs. neprofitabilă
  const profitDistributionData = useMemo(() => {
    if (!financialMetrics) return [];
    
    return [
      { name: 'Profitabile', value: financialMetrics.numarCaseProfitabile },
      { name: 'Neprofitabile', value: financialMetrics.numarCaseNeprofitabile }
    ];
  }, [financialMetrics]);
  
  if (!dateFinanciare.length || !latestYear) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <p>Nu există date financiare disponibile.</p>
      </div>
    );
  }
  
  return (
    <div className="financial-analysis">
      <div className="mb-4">
        <h2 className="text-xl font-bold mb-2">
          Analiză Financiară pentru Anul {latestYear}
        </h2>
        
        {financialMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <h3 className="text-lg font-medium text-blue-800">
                Profit Total
              </h3>
              <p className="text-2xl font-bold text-blue-600">
                {financialMetrics.profitTotal.toLocaleString()} RON
              </p>
              <p className="text-sm text-blue-500">
                {financialMetrics.numarCaseProfitabile} case profitabile
              </p>
            </div>
            
            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
              <h3 className="text-lg font-medium text-green-800">
                Contribuții Bugetare
              </h3>
              <p className="text-2xl font-bold text-green-600">
                {financialMetrics.contributiiBuget.toLocaleString()} RON
              </p>
              <p className="text-sm text-green-500">
                {financialMetrics.contributiiMediiPerAngajat.toLocaleString()} RON/angajat
              </p>
            </div>
            
            <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
              <h3 className="text-lg font-medium text-purple-800">
                Resurse Umane
              </h3>
              <p className="text-2xl font-bold text-purple-600">
                {financialMetrics.numarTotalAngajati.toLocaleString()} angajați
              </p>
              <p className="text-sm text-purple-500">
                {financialMetrics.fondSalariiTotal.toLocaleString()} RON fond salarii
              </p>
            </div>
          </div>
        )}
        
        <div className="mb-4 flex space-x-2 overflow-x-auto pb-2">
          <button
            className={`px-4 py-2 rounded ${
              activeView === 'profit' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
            onClick={() => setActiveView('profit')}
          >
            Profitabilitate
          </button>
          <button
            className={`px-4 py-2 rounded ${
              activeView === 'contributors' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
            onClick={() => setActiveView('contributors')}
          >
            Contribuții Buget
          </button>
          <button
            className={`px-4 py-2 rounded ${
              activeView === 'correlation' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
            onClick={() => setActiveView('correlation')}
          >
            Corelație Angajați-Profit
          </button>
          <button
            className={`px-4 py-2 rounded whitespace-nowrap ${
              activeView === 'data-table' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
            onClick={() => setActiveView('data-table')}
          >
            Date Financiare Detaliate
          </button>
        </div>
      </div>
      
      {activeView === 'profit' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Top 10 Case după Profit</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={profitabilityData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={150} />
                <Tooltip 
                  formatter={(value) => [`${value.toLocaleString()} RON`, 'Profit/Pierdere']}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white p-2 border rounded shadow">
                          <p className="text-sm font-semibold">{data.fullName}</p>
                          <p className="text-sm">{`${data.profit.toLocaleString()} RON`}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar 
                  dataKey="profit" 
                  fill={(entry) => entry.profit >= 0 ? "#4caf50" : "#f44336"}
                  name="Profit/Pierdere" 
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Distribuție Case Profitabile vs. Neprofitabile</h3>
            <ResponsiveContainer width="100%" height={400}>
              <PieChart>
                <Pie
                  data={profitDistributionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {profitDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? '#4caf50' : '#f44336'} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [value, 'Număr Case']} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      
      {activeView === 'contributors' && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Top 10 Contribuitori la Buget</h3>
          <ResponsiveContainer width="100%" height={500}>
            <BarChart data={budgetContributionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip 
                formatter={(value) => [`${value.toLocaleString()} RON`, 'Contribuție']}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-2 border rounded shadow">
                        <p className="text-sm font-semibold">{data.fullName}</p>
                        <p className="text-sm">{`${data.contributie.toLocaleString()} RON`}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend />
              <Bar dataKey="contributie" fill="#8884d8" name="Contribuție Buget" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      
      {activeView === 'correlation' && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Corelație Număr Angajați - Profit</h3>
          <ResponsiveContainer width="100%" height={500}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                dataKey="angajati" 
                name="Număr Angajați"
                domain={[0, 'dataMax']}
              />
              <YAxis 
                type="number" 
                dataKey="profit" 
                name="Profit/Pierdere"
              />
              <Tooltip 
                formatter={(value, name) => {
                  if (name === 'profit') return [`${value.toLocaleString()} RON`, 'Profit/Pierdere'];
                  if (name === 'angajati') return [value, 'Număr Angajați'];
                  return [value, name];
                }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-2 border rounded shadow">
                        <p className="text-sm font-semibold">{data.name}</p>
                        <p className="text-sm">{`Angajați: ${data.angajati}`}</p>
                        <p className="text-sm">{`Profit: ${data.profit.toLocaleString()} RON`}</p>
                        <p className="text-sm">{`Fond Salarii: ${data.fondSalarii.toLocaleString()} RON`}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend />
              <Scatter 
                name="Case de Schimb" 
                data={correlationData} 
                fill={(entry) => entry.profit >= 0 ? "#4caf50" : "#f44336"}
              />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-600 mt-2">
            Fiecare punct reprezintă o casă de schimb. Culorile indică profitabilitatea (verde = profit, roșu = pierdere).
          </p>
        </div>
      )}
      
      {/* Adăugăm noul tab pentru date financiare detaliate */}
      {activeView === 'data-table' && (
        <FinancialDataTable />
      )}
    </div>
  );
};

export default FinancialAnalysis;