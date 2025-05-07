// src/components/vizualizari/StatusChart.tsx
import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import useExchangeData from '../../hooks/useExchangeData';

const COLORS = ['#4caf50', '#f44336', '#ff9800'];

const StatusChart: React.FC = () => {
  const { puncteSchimb } = useExchangeData();
  
  const statusData = useMemo(() => {
    if (!puncteSchimb.length) return [];
    
    const counts = {
      active: 0,
      inactive: 0,
      unspecified: 0
    };
    
    puncteSchimb.forEach(punct => {
      if (!punct["DATA INCHIDERE PUNCT"]) {
        counts.active++;
      } else {
        counts.inactive++;
      }
    });
    
    return [
      { name: 'Active', value: counts.active },
      { name: 'Inactive', value: counts.inactive },
      { name: 'Nespecificat', value: counts.unspecified }
    ].filter(item => item.value > 0);
  }, [puncteSchimb]);
  
  if (!puncteSchimb.length) {
    return <div>Nu există date disponibile</div>;
  }
  
  return (
    <div className="status-chart">
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={statusData}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
          >
            {statusData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip 
            formatter={(value) => [value, 'Puncte de schimb']} 
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
      <div className="mt-4 text-center text-sm text-gray-600">
        <p>Total: {puncteSchimb.length} puncte de schimb</p>
      </div>
    </div>
  );
};

export default StatusChart;