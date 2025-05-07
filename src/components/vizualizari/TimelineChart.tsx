// src/components/vizualizari/TimelineChart.tsx
import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import useExchangeData from '../../hooks/useExchangeData';

const TimelineChart: React.FC = () => {
  const { caseSchimb, puncteSchimb } = useExchangeData();
  
  const timelineData = useMemo(() => {
    if (!caseSchimb.length || !puncteSchimb.length) return [];
    
    // Obține anii unici de autorizare
    const years = new Set<number>();
    
    caseSchimb.forEach(casa => {
      if (casa["DATA AUTORIZARE CASA"]) {
        const year = new Date(casa["DATA AUTORIZARE CASA"]).getFullYear();
        if (year > 1990 && year < 2030) { // Validare de bază pentru ani rezonabili
          years.add(year);
        }
      }
    });
    
    puncteSchimb.forEach(punct => {
      if (punct["DATA AUTORIZARE PUNCT"]) {
        const year = new Date(punct["DATA AUTORIZARE PUNCT"]).getFullYear();
        if (year > 1990 && year < 2030) {
          years.add(year);
        }
      }
    });
    
    // Sortează anii
    const sortedYears = Array.from(years).sort();
    
    // Calculează cumulativ numărul de case și puncte pentru fiecare an
    return sortedYears.map(year => {
      const casePanaLaAn = caseSchimb.filter(casa => {
        if (!casa["DATA AUTORIZARE CASA"]) return false;
        const casaYear = new Date(casa["DATA AUTORIZARE CASA"]).getFullYear();
        return casaYear <= year;
      }).length;
      
      const punctePanaLaAn = puncteSchimb.filter(punct => {
        if (!punct["DATA AUTORIZARE PUNCT"]) return false;
        const punctYear = new Date(punct["DATA AUTORIZARE PUNCT"]).getFullYear();
        return punctYear <= year;
      }).length;
      
      return {
        year,
        case: casePanaLaAn,
        puncte: punctePanaLaAn
      };
    });
  }, [caseSchimb, puncteSchimb]);
  
  if (!timelineData.length) {
    return <div>Nu există date disponibile pentru cronologie</div>;
  }
  
  return (
    <div className="timeline-chart">
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={timelineData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="case" 
            stroke="#8884d8" 
            name="Case de schimb" 
            activeDot={{ r: 8 }} 
          />
          <Line 
            type="monotone" 
            dataKey="puncte" 
            stroke="#82ca9d" 
            name="Puncte de schimb" 
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-4 text-center text-sm text-gray-600">
        <p>Evoluția numărului de case și puncte de schimb în timp</p>
      </div>
    </div>
  );
};

export default TimelineChart;