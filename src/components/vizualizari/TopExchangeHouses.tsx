// src/components/vizualizari/TopExchangeHouses.tsx
import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import useExchangeData from '../../hooks/useExchangeData';

const TopExchangeHouses: React.FC = () => {
  const { caseSchimb, puncteSchimb } = useExchangeData();
  
  const topHousesData = useMemo(() => {
    if (!caseSchimb.length || !puncteSchimb.length) return [];
    
    // Calculează numărul de puncte pentru fiecare casă
    const housesWithPoints = caseSchimb.map(casa => {
      const count = puncteSchimb.filter(punct => 
        punct["COD CASA  CU S"] === casa["COD CASA  CU S"]
      ).length;
      
      return {
        id: casa["COD CASA  CU S"],
        name: casa["DENUMIRE CASA"]?.length > 20 
          ? casa["DENUMIRE CASA"].substring(0, 17) + '...'
          : casa["DENUMIRE CASA"],
        fullName: casa["DENUMIRE CASA"],
        count
      };
    });
    
    // Sortează și ia primele 10
    return housesWithPoints
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }, [caseSchimb, puncteSchimb]);
  
  if (!topHousesData.length) {
    return <div>Nu există date disponibile</div>;
  }
  
  return (
    <div className="top-exchange-houses">
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={topHousesData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip 
            formatter={(value) => [value, 'Puncte de schimb']}
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                return (
                  <div className="bg-white p-2 border rounded shadow">
                    <p className="text-sm font-semibold">{data.fullName}</p>
                    <p className="text-sm">{`${data.count} puncte de schimb`}</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="count" fill="#8884d8" name="Număr puncte" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TopExchangeHouses;