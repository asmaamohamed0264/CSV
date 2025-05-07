// src/components/vizualizari/BucharestSectorsMap.tsx
import React, { useState, useMemo } from 'react';
import { 
  ComposableMap, 
  Geographies, 
  Geography,
  ZoomableGroup
} from 'react-simple-maps';
import { scaleQuantize } from 'd3-scale';
import { Tooltip } from 'react-tooltip';
import useExchangeData from '../../hooks/useExchangeData';
import { 
  normalizeLocationName, 
  isBucharestSector, 
  extractSectorNumber 
} from '../../utils/normalizationUtils';

// Calea către fișierul TopoJSON cu sectoarele București
// Acest fișier trebuie creat și adăugat în proiect
const BUCHAREST_SECTORS_TOPO_JSON = '/topojson/bucharest-sectors.json';

interface SectorData {
  sector: string;
  count: number;
  color: string;
}

const BucharestSectorsMap: React.FC = () => {
  // Utilizarea hook-ului pentru date
  const { 
    filteredPuncteSchimb, 
    isLoading, 
    hasError, 
    updateFilters 
  } = useExchangeData();
  
  // State pentru sectorul selectat
  const [selectedSector, setSelectedSector] = useState<string | null>(null);
  // State pentru datele despre sectorul sub cursor
  const [tooltipContent, setTooltipContent] = useState<SectorData | null>(null);
  
  // Calculăm distribuția pe sectoare
  const sectorDistribution = useMemo(() => {
    const distribution: Record<string, number> = {};
    
    // Inițializăm toate sectoarele cu 0
    for (let i = 1; i <= 6; i++) {
      distribution[`București Sectorul ${i}`] = 0;
    }
    
    filteredPuncteSchimb.forEach(punct => {
      if (punct.judet) {
        const location = normalizeLocationName(punct.judet);
        
        // Verificăm dacă este un sector din București
        if (isBucharestSector(location)) {
          distribution[location] = (distribution[location] || 0) + 1;
        } else if (location === 'București') {
          // Pentru cazurile în care apare doar "București" fără sector specificat
          // Adăugăm la un total separat sau distribuim în mod egal
          // Aici am ales să adăugăm la un total separat "București (nespecificat)"
          distribution['București (nespecificat)'] = (distribution['București (nespecificat)'] || 0) + 1;
        }
      }
    });
    
    return distribution;
  }, [filteredPuncteSchimb]);
  
  // Găsim numărul maxim de puncte pentru un sector pentru a scala culorile
  const maxCount = useMemo(() => {
    return Math.max(...Object.values(sectorDistribution), 1);
  }, [sectorDistribution]);
  
  // Scala de culori pentru sectoare în funcție de numărul de puncte
  const colorScale = scaleQuantize<string>()
    .domain([0, maxCount])
    .range([
      "#e6f2ff", "#cce6ff", "#99c2ff", "#4d94ff", 
      "#1a75ff", "#0066ff", "#0047b3", "#003380"
    ]);
  
  // Handler pentru click pe un sector
  const handleSectorClick = (sector: string) => {
    // Dacă sectorul este deja selectat, anulăm selecția
    if (selectedSector === sector) {
      setSelectedSector(null);
      updateFilters({ judet: '' });
    } else {
      setSelectedSector(sector);
      updateFilters({ judet: sector });
    }
  };
  
  // Handler pentru hover pe un sector
  const handleSectorHover = (sector: string) => {
    const fullSectorName = `București Sectorul ${sector}`;
    const count = sectorDistribution[fullSectorName] || 0;
    const color = count > 0 ? colorScale(count) : "#f3f4f6";
    
    setTooltipContent({ 
      sector: fullSectorName, 
      count, 
      color 
    });
  };
  
  // Handler pentru ieșirea cursorului de pe un sector
  const handleSectorLeave = () => {
    setTooltipContent(null);
  };
  
  // Date pentru graficul de bare pentru comparație
  const barChartData = useMemo(() => {
    return Object.entries(sectorDistribution)
      .filter(([key]) => key !== 'București (nespecificat)')
      .map(([sector, count]) => ({
        sector: extractSectorNumber(sector),
        count,
      }))
      .sort((a, b) => (a.sector || 0) - (b.sector || 0));
  }, [sectorDistribution]);
  
  // Randarea componentei
  return (
    <div className="bucharest-sectors-map">
      <h2 className="text-xl font-bold mb-4">
        Distribuția Punctelor de Schimb Valutar în București pe Sectoare
      </h2>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : hasError ? (
        <div className="text-red-500 p-4 border border-red-300 rounded bg-red-50">
          A apărut o eroare la încărcarea datelor. 
          Vă rugăm să reîncărcați pagina sau să încercați mai târziu.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Harta sectoarelor */}
          <div className="relative">
            <ComposableMap
              projection="geoMercator"
              projectionConfig={{
                scale: 60000,
                center: [26.1, 44.43] // Coordonatele aproximative pentru centrul Bucureștiului
              }}
              width={400}
              height={400}
              data-tooltip-id="sector-tooltip"
            >
              <ZoomableGroup>
                <Geographies geography={BUCHAREST_SECTORS_TOPO_JSON}>
                  {({ geographies }) =>
                    geographies.map((geo) => {
                      const sectorNumber = geo.properties.sector;
                      const sectorName = `București Sectorul ${sectorNumber}`;
                      const count = sectorDistribution[sectorName] || 0;
                      const color = count > 0 ? colorScale(count) : "#f3f4f6";
                      
                      return (
                        <Geography
                          key={geo.rsmKey}
                          geography={geo}
                          fill={selectedSector === sectorName ? "#fbbf24" : color}
                          stroke="#999"
                          strokeWidth={0.5}
                          style={{
                            default: { outline: "none" },
                            hover: { outline: "none", fill: "#fbbf24" },
                            pressed: { outline: "none" }
                          }}
                          onClick={() => handleSectorClick(sectorName)}
                          onMouseEnter={() => handleSectorHover(sectorNumber)}
                          onMouseLeave={handleSectorLeave}
                        />
                      );
                    })
                  }
                </Geographies>
              </ZoomableGroup>
            </ComposableMap>
            
            <Tooltip id="sector-tooltip">
              {tooltipContent && (
                <div className="p-2">
                  <div className="font-bold">{tooltipContent.sector}</div>
                  <div className="text-sm">
                    {tooltipContent.count} {tooltipContent.count === 1 
                      ? 'punct de schimb' 
                      : 'puncte de schimb'}
                  </div>
                </div>
              )}
            </Tooltip>
            
            {/* Legendă pentru culori */}
            <div className="absolute bottom-2 left-2 bg-white p-2 rounded shadow-md text-xs">
              <div className="font-bold mb-1">Număr puncte de schimb:</div>
              <div className="flex items-center">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="flex flex-col items-center mr-2">
                    <div 
                      className="w-3 h-3 mb-1" 
                      style={{ backgroundColor: colorScale((i / 4) * maxCount) }}
                    ></div>
                    <span>{Math.round((i / 4) * maxCount)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Grafic bare pentru comparație */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-3">Distribuție pe Sectoare</h3>
            
            <div className="flex flex-col space-y-2 mb-4">
              {barChartData.map(({ sector, count }) => (
                <div key={sector} className="flex items-center">
                  <div className="w-16 text-right pr-2">Sector {sector}</div>
                  <div 
                    className="h-6 bg-blue-500 rounded"
                    style={{ 
                      width: `${(count / maxCount) * 100}%`,
                      minWidth: count > 0 ? '10px' : '0'
                    }}
                  ></div>
                  <div className="ml-2">{count}</div>
                </div>
              ))}
              
              {/* Afișează și punctele nespecificate dacă există */}
              {sectorDistribution['București (nespecificat)'] > 0 && (
                <div className="flex items-center">
                  <div className="w-16 text-right pr-2 text-xs">Nespecificat</div>
                  <div 
                    className="h-6 bg-gray-400 rounded"
                    style={{ 
                      width: `${(sectorDistribution['București (nespecificat)'] / maxCount) * 100}%`,
                      minWidth: '10px'
                    }}
                  ></div>
                  <div className="ml-2">{sectorDistribution['București (nespecificat)']}</div>
                </div>
              )}
            </div>
            
            <div className="text-sm text-gray-600">
              <p>Total puncte în București: {Object.values(sectorDistribution).reduce((a, b) => a + b, 0)}</p>
              <p className="mt-1">
                Sector cu cele mai multe puncte: {
                  barChartData.length > 0 
                    ? `Sectorul ${barChartData.reduce((max, current) => 
                        current.count > max.count ? current : max
                      ).sector} (${barChartData.reduce((max, current) => 
                        current.count > max.count ? current : max
                      ).count} puncte)`
                    : 'Niciunul'
                }
              </p>
            </div>
          </div>
        </div>
      )}
      
      {/* Informații suplimentare pentru sectorul selectat */}
      {selectedSector && (
        <div className="mt-4 p-4 border rounded bg-yellow-50">
          <h3 className="font-bold text-lg mb-2">
            {selectedSector}: {sectorDistribution[selectedSector] || 0} puncte de schimb
          </h3>
          
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={() => {
              setSelectedSector(null);
              updateFilters({ judet: '' });
            }}
          >
            Reset Selecție
          </button>
        </div>
      )}
    </div>
  );
};

export default BucharestSectorsMap;
