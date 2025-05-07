// src/components/vizualizari/CountyDistributionMap.tsx
import React, { useState, useEffect, useMemo } from 'react';
import { 
  ComposableMap, 
  Geographies, 
  Geography,
  ZoomableGroup
} from 'react-simple-maps';
import { scaleQuantize } from 'd3-scale';
import { Tooltip } from 'react-tooltip';
import useExchangeData from '../../hooks/useExchangeData';
import { normalizeLocationName } from '../../utils/normalizationUtils';

// Calea către fișierul TopoJSON cu județele României
// Acest fișier trebuie creat și adăugat în proiect
const ROMANIA_TOPO_JSON = '/topojson/romania-counties.json';

// Tipul pentru proprietățile unei regiuni pe hartă
interface RegionData {
  county: string;
  count: number;
  color: string;
}

// Componenta principală pentru harta distribuției
const CountyDistributionMap: React.FC = () => {
  // Utilizarea hook-ului pentru date
  const { 
    filteredPuncteSchimb, 
    isLoading, 
    hasError, 
    updateFilters 
  } = useExchangeData();
  
  // State pentru regiunea selectată
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  // State pentru datele despre regiunea sub cursor
  const [tooltipContent, setTooltipContent] = useState<RegionData | null>(null);
  
  // Calculăm distribuția pe județe
  const countyDistribution = useMemo(() => {
    const distribution: Record<string, number> = {};
    
    filteredPuncteSchimb.forEach(punct => {
      if (punct.judet) {
        const county = normalizeLocationName(punct.judet);
        
        // Ignoră sectoarele București pentru această vizualizare
        // Le vom agrega toate sub "București"
        const normalizedCounty = county.startsWith('București Sectorul') 
          ? 'București' 
          : county;
        
        if (!distribution[normalizedCounty]) {
          distribution[normalizedCounty] = 0;
        }
        
        distribution[normalizedCounty]++;
      }
    });
    
    return distribution;
  }, [filteredPuncteSchimb]);
  
  // Găsim numărul maxim de puncte pentru un județ pentru a scala culorile
  const maxCount = useMemo(() => {
    return Math.max(...Object.values(countyDistribution), 1);
  }, [countyDistribution]);
  
  // Scala de culori pentru județe în funcție de numărul de puncte
  const colorScale = scaleQuantize<string>()
    .domain([0, maxCount])
    .range([
      "#e6f2ff", "#cce6ff", "#99c2ff", "#4d94ff", 
      "#1a75ff", "#0066ff", "#0047b3", "#003380"
    ]);
  
  // Handler pentru click pe un județ
  const handleCountyClick = (county: string) => {
    // Dacă județul este deja selectat, anulăm selecția
    if (selectedRegion === county) {
      setSelectedRegion(null);
      updateFilters({ judet: '' });
    } else {
      setSelectedRegion(county);
      updateFilters({ judet: county });
    }
  };
  
  // Handler pentru hover pe un județ
  const handleCountyHover = (county: string, count: number, color: string) => {
    setTooltipContent({ county, count, color });
  };
  
  // Handler pentru ieșirea cursorului de pe un județ
  const handleCountyLeave = () => {
    setTooltipContent(null);
  };
  
  // Randarea componentei
  return (
    <div className="county-distribution-map">
      <h2 className="text-xl font-bold mb-4">
        Distribuția Punctelor de Schimb Valutar pe Județe
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
        <div className="relative">
          <ComposableMap
            projection="geoMercator"
            projectionConfig={{
              scale: 6000,
              center: [25, 46] // Coordonatele aproximative pentru centrul României
            }}
            width={800}
            height={500}
            data-tooltip-id="county-tooltip"
          >
            <ZoomableGroup>
              <Geographies geography={ROMANIA_TOPO_JSON}>
                {({ geographies }) =>
                  geographies.map((geo) => {
                    const countyName = geo.properties.name;
                    const count = countyDistribution[countyName] || 0;
                    const color = count > 0 ? colorScale(count) : "#f3f4f6";
                    
                    return (
                      <Geography
                        key={geo.rsmKey}
                        geography={geo}
                        fill={selectedRegion === countyName ? "#fbbf24" : color}
                        stroke="#999"
                        strokeWidth={0.5}
                        style={{
                          default: { outline: "none" },
                          hover: { outline: "none", fill: "#fbbf24" },
                          pressed: { outline: "none" }
                        }}
                        onClick={() => handleCountyClick(countyName)}
                        onMouseEnter={() => handleCountyHover(countyName, count, color)}
                        onMouseLeave={handleCountyLeave}
                      />
                    );
                  })
                }
              </Geographies>
            </ZoomableGroup>
          </ComposableMap>
          
          <Tooltip id="county-tooltip">
            {tooltipContent && (
              <div className="p-2">
                <div className="font-bold">{tooltipContent.county}</div>
                <div className="text-sm">
                  {tooltipContent.count} {tooltipContent.count === 1 
                    ? 'punct de schimb' 
                    : 'puncte de schimb'}
                </div>
              </div>
            )}
          </Tooltip>
          
          {/* Legendă pentru culori */}
          <div className="absolute bottom-2 left-2 bg-white p-2 rounded shadow-md text-sm">
            <div className="font-bold mb-1">Număr puncte de schimb:</div>
            <div className="flex items-center">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="flex flex-col items-center mr-2">
                  <div 
                    className="w-4 h-4 mb-1" 
                    style={{ backgroundColor: colorScale((i / 7) * maxCount) }}
                  ></div>
                  <span>{Math.round((i / 7) * maxCount)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      
      {/* Informații suplimentare pentru județul selectat */}
      {selectedRegion && (
        <div className="mt-4 p-4 border rounded bg-yellow-50">
          <h3 className="font-bold text-lg mb-2">
            {selectedRegion}: {countyDistribution[selectedRegion] || 0} puncte de schimb
          </h3>
          
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={() => {
              setSelectedRegion(null);
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

export default CountyDistributionMap;
