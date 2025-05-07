// src/components/analiza/DataAnalysisLLM.tsx
import React, { useState, useEffect } from 'react';
import { useExchangeData } from '../../hooks/useExchangeData';
import LLMService from '../../services/llmService';

// Tipuri de analize care pot fi solicitate
type AnalysisType = 
  | 'trend' 
  | 'geographic' 
  | 'financial' 
  | 'custom';

interface Props {
  initialQuestion?: string;
}

const DataAnalysisLLM: React.FC<Props> = ({ initialQuestion = '' }) => {
  const { 
    caseSchimb, 
    puncteSchimb, 
    dateFinanciare, 
    distributionByLocation,
    financialMetrics
  } = useExchangeData();
  
  const [question, setQuestion] = useState(initialQuestion);
  const [analysis, setAnalysis] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedAnalysisType, setSelectedAnalysisType] = useState<AnalysisType>('trend');
  const [insights, setInsights] = useState<string[]>([]);
  
  // Generare analiză la schimbarea tipului sau la încărcarea inițială (dacă există întrebare)
  useEffect(() => {
    if (initialQuestion) {
      handleQuestionSubmit();
    }
  }, [initialQuestion]);
  
  // Generare de insights automate la încărcarea componentei
  useEffect(() => {
    if (caseSchimb.length && puncteSchimb.length && dateFinanciare.length) {
      generateInsights();
    }
  }, [caseSchimb, puncteSchimb, dateFinanciare]);
  
  // Generează insights automate
  const generateInsights = async () => {
    setIsLoading(true);
    
    try {
      // Pregătește un context condensat pentru LLM
      const dataContext = {
        numCaseSchimb: caseSchimb.length,
        numPuncteSchimb: puncteSchimb.length,
        topJudete: distributionByLocation.slice(0, 5),
        metrics: financialMetrics,
        trends: {
          caseActive: caseSchimb.filter(c => !c["DATA INCHIDERE CASA"]).length,
          puncteActive: puncteSchimb.filter(p => !p["DATA INCHIDERE PUNCT"]).length,
          profitTotal: financialMetrics?.profitTotal || 0,
          numarAngajati: financialMetrics?.numarTotalAngajati || 0
        }
      };
      
      // Obține insights de la LLM
      const insightsList = await LLMService.suggestInsights(dataContext);
      setInsights(insightsList);
    } catch (error) {
      console.error('Eroare la generarea insights-urilor:', error);
      setInsights(['Nu s-au putut genera insights automate.']);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Trimite întrebarea către LLM
  const handleQuestionSubmit = async () => {
    if (!question.trim()) return;
    
    setIsLoading(true);
    setAnalysis('');
    
    try {
      // Pregătește contextul pentru LLM
      let dataContext: any = {
        numCaseSchimb: caseSchimb.length,
        numPuncteSchimb: puncteSchimb.length,
        distributionByLocation: distributionByLocation.slice(0, 10),
        financialMetrics
      };
      
      // Adaugă contextul specific în funcție de tipul de analiză
      switch (selectedAnalysisType) {
        case 'trend':
          dataContext.trends = {
            last3Years: dateFinanciare
              .filter(d => d["ARE_BILANT_DA/NU"] === "DA")
              .sort((a, b) => b["AN DE REFERINTA"] - a["AN DE REFERINTA"])
              .slice(0, 3)
              .map(year => ({
                an: year["AN DE REFERINTA"],
                totalProfit: dateFinanciare
                  .filter(d => d["AN DE REFERINTA"] === year["AN DE REFERINTA"])
                  .reduce((sum, item) => {
                    const profit = typeof item["PROFIT_PIERDERE"] === 'string'
                      ? parseFloat(item["PROFIT_PIERDERE"])
                      : item["PROFIT_PIERDERE"] || 0;
                    return sum + profit;
                  }, 0),
                totalContributii: dateFinanciare
                  .filter(d => d["AN DE REFERINTA"] === year["AN DE REFERINTA"])
                  .reduce((sum, item) => {
                    const contributii = typeof item["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] === 'string'
                      ? parseFloat(item["INCASARI LA BUGETUL GENESAL CONSOLIDAT"])
                      : item["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] || 0;
                    return sum + contributii;
                  }, 0)
              }))
          };
          break;
          
        case 'geographic':
          dataContext.geografic = {
            topJudete: distributionByLocation.slice(0, 10),
            distributionDetails: distributionByLocation.map(loc => {
              const casaIds = puncteSchimb
                .filter(p => p["JUDET / SECTOR PUNCT"] === loc.judet)
                .map(p => p["COD CASA  CU S"]);
              
              const uniqueCasaIds = [...new Set(casaIds)];
              
              return {
                judet: loc.judet,
                puncte: loc.count,
                case: uniqueCasaIds.length
              };
            })
          };
          break;
          
        case 'financial':
          // Adaugă date pentru top case după profit
          dataContext.financial = {
            topProfit: dateFinanciare
              .filter(d => d["ARE_BILANT_DA/NU"] === "DA" && d["PROFIT_PIERDERE"] !== "")
              .map(d => ({
                denumire: d["DENUMIRE"],
                profit: typeof d["PROFIT_PIERDERE"] === 'string'
                  ? parseFloat(d["PROFIT_PIERDERE"])
                  : d["PROFIT_PIERDERE"],
                angajati: typeof d["NR_MEDIU SALARIATI"] === 'string'
                  ? parseInt(d["NR_MEDIU SALARIATI"], 10)
                  : d["NR_MEDIU SALARIATI"]
              }))
              .sort((a, b) => b.profit - a.profit)
              .slice(0, 5),
              
            // Adaugă date pentru case cu cele mai multe puncte
            topCompaniiDupaPuncte: Object.entries(
              puncteSchimb.reduce((acc, punct) => {
                const casaId = punct["COD CASA  CU S"];
                acc[casaId] = (acc[casaId] || 0) + 1;
                return acc;
              }, {} as Record<string, number>)
            )
              .map(([casaId, count]) => {
                const casa = caseSchimb.find(c => c["COD CASA  CU S"] === casaId);
                return {
                  denumire: casa ? casa["DENUMIRE CASA"] : 'Necunoscut',
                  puncte: count
                };
              })
              .sort((a, b) => b.puncte - a.puncte)
              .slice(0, 5)
          };
          break;
          
        case 'custom':
          // Pentru întrebări custom, includem toate datele relevante
          dataContext = {
            ...dataContext,
            caseleMari: caseSchimb
              .filter(casa => {
                const numarPuncte = puncteSchimb.filter(p => 
                  p["COD CASA  CU S"] === casa["COD CASA  CU S"]
                ).length;
                return numarPuncte >= 5;
              })
              .map(casa => ({
                denumire: casa["DENUMIRE CASA"],
                codFiscal: casa["COD FISCAL CASA"],
                puncte: puncteSchimb.filter(p => 
                  p["COD CASA  CU S"] === casa["COD CASA  CU S"]
                ).length,
                active: !casa["DATA INCHIDERE CASA"]
              })),
              
            topDate: {
              ceaMaiVecheData: caseSchimb
                .map(casa => new Date(casa["DATA AUTORIZARE CASA"]).getTime())
                .sort()[0],
              ceaMaiRecentaData: caseSchimb
                .map(casa => new Date(casa["DATA AUTORIZARE CASA"]).getTime())
                .sort((a, b) => b - a)[0]
            }
          };
          break;
      }
      
      // Trimite întrebarea către LLM
      const response = await LLMService.interpretQuestion(question, dataContext);
      setAnalysis(response);
    } catch (error) {
      console.error('Eroare la interpretarea întrebării:', error);
      setAnalysis('Nu s-a putut genera o analiză. Vă rugăm să încercați din nou mai târziu.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Generează un raport narativ
  const generateNarrativeReport = async () => {
    setIsLoading(true);
    setAnalysis('');
    
    try {
      // Pregătește contextul pentru raport
      const dataContext = {
        numCaseSchimb: caseSchimb.length,
        numPuncteSchimb: puncteSchimb.length,
        caseActive: caseSchimb.filter(c => !c["DATA INCHIDERE CASA"]).length,
        puncteActive: puncteSchimb.filter(p => !p["DATA INCHIDERE PUNCT"]).length,
        distributieJudete: distributionByLocation.slice(0, 10),
        topCase: Object.entries(
          puncteSchimb.reduce((acc, punct) => {
            const casaId = punct["COD CASA  CU S"];
            acc[casaId] = (acc[casaId] || 0) + 1;
            return acc;
          }, {} as Record<string, number>)
        )
          .map(([casaId, count]) => {
            const casa = caseSchimb.find(c => c["COD CASA  CU S"] === casaId);
            return {
              denumire: casa ? casa["DENUMIRE CASA"] : 'Necunoscut',
              puncte: count
            };
          })
          .sort((a, b) => b.puncte - a.puncte)
          .slice(0, 5),
        financialMetrics,
        topProfit: dateFinanciare
          .filter(d => d["ARE_BILANT_DA/NU"] === "DA" && d["PROFIT_PIERDERE"] !== "")
          .map(d => ({
            denumire: d["DENUMIRE"],
            profit: typeof d["PROFIT_PIERDERE"] === 'string'
              ? parseFloat(d["PROFIT_PIERDERE"])
              : d["PROFIT_PIERDERE"]
          }))
          .sort((a, b) => b.profit - a.profit)
          .slice(0, 5)
      };
      
      // Prompt pentru raport
      const promptTemplate = `
        Generează un raport narativ concis despre casele de schimb valutar din România,
        bazat pe datele furnizate. Raportul ar trebui să includă:
        
        1. O introducere generală despre piața caselor de schimb
        2. Distribuția geografică a punctelor de schimb
        3. Performanța financiară și contribuțiile la buget
        4. Tendințe principale observate
        5. Concluzii
        
        Folosește un ton profesional și obiectiv, bazat strict pe datele furnizate.
      `;
      
      // Generează raportul
      const report = await LLMService.generateReport(dataContext, promptTemplate);
      setAnalysis(report);
    } catch (error) {
      console.error('Eroare la generarea raportului:', error);
      setAnalysis('Nu s-a putut genera raportul. Vă rugăm să încercați din nou mai târziu.');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="data-analysis-llm bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Analiză Asistată de AI</h2>
      
      {/* Afișare insights automate */}
      {insights.length > 0 && (
        <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h3 className="text-lg font-semibold text-blue-800 mb-2">Insights Detectate</h3>
          <ul className="list-disc pl-5 space-y-1">
            {insights.map((insight, index) => (
              <li key={index} className="text-blue-700">{insight}</li>
            ))}
          </ul>
        </div>
      )}
      
      {/* Selectare tip analiză */}
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">Tipul Analizei</h3>
        <div className="flex flex-wrap gap-2">
          <button
            className={`px-3 py-1 rounded ${
              selectedAnalysisType === 'trend' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
            onClick={() => setSelectedAnalysisType('trend')}
          >
            Tendințe
          </button>
          <button
            className={`px-3 py-1 rounded ${
              selectedAnalysisType === 'geographic' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
            onClick={() => setSelectedAnalysisType('geographic')}
          >
            Distribuție Geografică
          </button>
          <button
            className={`px-3 py-1 rounded ${
              selectedAnalysisType === 'financial' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
            onClick={() => setSelectedAnalysisType('financial')}
          >
            Performanță Financiară
          </button>
          <button
            className={`px-3 py-1 rounded ${
              selectedAnalysisType === 'custom' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}
            onClick={() => setSelectedAnalysisType('custom')}
          >
            Întrebare Personalizată
          </button>
        </div>
      </div>
      
      {/* Sugestii de întrebări predefinite */}
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">Întrebări Sugerate</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {selectedAnalysisType === 'trend' && (
            <>
              <button 
                className="text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
                onClick={() => {
                  setQuestion("Cum a evoluat profitabilitatea caselor de schimb în ultimii ani?");
                  handleQuestionSubmit();
                }}
              >
                Cum a evoluat profitabilitatea caselor de schimb în ultimii ani?
              </button>
              <button 
                className="text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
                onClick={() => {
                  setQuestion("Care sunt tendințele în contribuțiile la buget ale caselor de schimb?");
                  handleQuestionSubmit();
                }}
              >
                Care sunt tendințele în contribuțiile la buget ale caselor de schimb?
              </button>
            </>
          )}
          
          {selectedAnalysisType === 'geographic' && (
            <>
              <button 
                className="text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
                onClick={() => {
                  setQuestion("Care sunt județele cu cele mai multe puncte de schimb valutar?");
                  handleQuestionSubmit();
                }}
              >
                Care sunt județele cu cele mai multe puncte de schimb valutar?
              </button>
              <button 
                className="text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
                onClick={() => {
                  setQuestion("Cum sunt distribuite casele de schimb între regiunile țării?");
                  handleQuestionSubmit();
                }}
              >
                Cum sunt distribuite casele de schimb între regiunile țării?
              </button>
            </>
          )}
          
          {selectedAnalysisType === 'financial' && (
            <>
              <button 
                className="text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
                onClick={() => {
                  setQuestion("Care sunt cele mai profitabile case de schimb?");
                  handleQuestionSubmit();
                }}
              >
                Care sunt cele mai profitabile case de schimb?
              </button>
              <button 
                className="text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
                onClick={() => {
                  setQuestion("Există o corelație între numărul de angajați și profitabilitate?");
                  handleQuestionSubmit();
                }}
              >
                Există o corelație între numărul de angajați și profitabilitate?
              </button>
            </>
          )}
          
          {selectedAnalysisType === 'custom' && (
            <>
              <button 
                className="text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
                onClick={() => {
                  setQuestion("Care sunt cele mai importante case de schimb după numărul de puncte?");
                  handleQuestionSubmit();
                }}
              >
                Care sunt cele mai importante case de schimb după numărul de puncte?
              </button>
              <button 
                className="text-left px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
                onClick={() => {
                  setQuestion("Generează un raport comprehensiv despre piața caselor de schimb valutar");
                  generateNarrativeReport();
                }}
              >
                Generează un raport comprehensiv despre piața caselor de schimb valutar
              </button>
            </>
          )}
        </div>
      </div>
      
      {/* Formular pentru întrebare personalizată */}
      <div className="mb-4">
        <div className="flex">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Adresează o întrebare despre datele disponibile..."
            className="flex-grow px-4 py-2 border rounded-l focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleQuestionSubmit}
            disabled={isLoading || !question.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-r hover:bg-blue-600 disabled:bg-gray-400"
          >
            {isLoading ? 'Se procesează...' : 'Analizează'}
          </button>
        </div>
      </div>
      
      {/* Afișare răspuns */}
      {analysis && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border">
          <h3 className="text-lg font-semibold mb-2">Analiză</h3>
          <div className="prose max-w-none">
            {analysis.split('\n\n').map((paragraph, index) => (
              <p key={index} className="mb-2">{paragraph}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DataAnalysisLLM;
