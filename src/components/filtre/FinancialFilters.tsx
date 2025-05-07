// src/components/filtre/FinancialFilters.tsx
import React, { useState, useEffect } from 'react';
import { useExchangeData } from '../../hooks/useExchangeData';

interface RangeFilter {
  min: string;
  max: string;
}

interface FinancialFiltersProps {
  className?: string;
  onApplyFilters: (filters: any) => void;
}

const FinancialFilters: React.FC<FinancialFiltersProps> = ({ className = '', onApplyFilters }) => {
  const { dateFinanciare } = useExchangeData();
  
  // State pentru filtre
  const [profitRange, setProfitRange] = useState<RangeFilter>({ min: '', max: '' });
  const [contributiiRange, setContributiiRange] = useState<RangeFilter>({ min: '', max: '' });
  const [angajatiRange, setAngajatiRange] = useState<RangeFilter>({ min: '', max: '' });
  const [anReferinta, setAnReferinta] = useState<string>('');
  const [areBilant, setAreBilant] = useState<string>('toate');
  const [sortBy, setSortBy] = useState<string>('');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  
  // Extragere listă de ani unici din date
  const uniqueYears = React.useMemo(() => {
    if (!dateFinanciare || !dateFinanciare.length) return [];
    
    const years = new Set<number>();
    
    dateFinanciare.forEach(data => {
      if (data["AN DE REFERINTA"]) {
        years.add(data["AN DE REFERINTA"]);
      }
    });
    
    return Array.from(years).sort((a, b) => b - a);
  }, [dateFinanciare]);
  
  // Handler pentru aplicarea filtrelor
  const handleApplyFilters = () => {
    const filters = {
      financialFilters: {
        profitMin: profitRange.min !== '' ? parseFloat(profitRange.min) : undefined,
        profitMax: profitRange.max !== '' ? parseFloat(profitRange.max) : undefined,
        contributiiMin: contributiiRange.min !== '' ? parseFloat(contributiiRange.min) : undefined,
        contributiiMax: contributiiRange.max !== '' ? parseFloat(contributiiRange.max) : undefined,
        angajatiMin: angajatiRange.min !== '' ? parseInt(angajatiRange.min, 10) : undefined,
        angajatiMax: angajatiRange.max !== '' ? parseInt(angajatiRange.max, 10) : undefined,
        anReferinta: anReferinta !== '' ? parseInt(anReferinta, 10) : undefined,
        areBilant: areBilant !== 'toate' ? areBilant === 'da' : undefined,
        sortBy,
        sortDirection
      }
    };
    
    onApplyFilters(filters);
  };
  
  // Handler pentru resetarea filtrelor
  const handleResetFilters = () => {
    setProfitRange({ min: '', max: '' });
    setContributiiRange({ min: '', max: '' });
    setAngajatiRange({ min: '', max: '' });
    setAnReferinta('');
    setAreBilant('toate');
    setSortBy('');
    setSortDirection('desc');
    
    onApplyFilters({ financialFilters: {} });
  };
  
  // Setarea anului cel mai recent la încărcare
  useEffect(() => {
    if (uniqueYears.length > 0 && !anReferinta) {
      setAnReferinta(uniqueYears[0].toString());
    }
  }, [uniqueYears, anReferinta]);
  
  return (
    <div className={`financial-filters p-4 bg-white rounded-lg shadow ${className}`}>
      <h3 className="text-lg font-semibold mb-4">Filtre Avansate Date Financiare</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* An de referință */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            An de referință
          </label>
          <select
            value={anReferinta}
            onChange={(e) => setAnReferinta(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
          >
            <option value="">Toți anii</option>
            {uniqueYears.map(year => (
              <option key={year} value={year}>{year}</option>
            ))}
          </select>
        </div>
        
        {/* Are bilanț */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Are bilanț
          </label>
          <div className="flex space-x-4 mt-2">
            <label className="inline-flex items-center">
              <input
                type="radio"
                className="form-radio text-blue-600"
                name="areBilant"
                value="toate"
                checked={areBilant === 'toate'}
                onChange={() => setAreBilant('toate')}
              />
              <span className="ml-2">Toate</span>
            </label>
            <label className="inline-flex items-center">
              <input
                type="radio"
                className="form-radio text-blue-600"
                name="areBilant"
                value="da"
                checked={areBilant === 'da'}
                onChange={() => setAreBilant('da')}
              />
              <span className="ml-2">Da</span>
            </label>
            <label className="inline-flex items-center">
              <input
                type="radio"
                className="form-radio text-blue-600"
                name="areBilant"
                value="nu"
                checked={areBilant === 'nu'}
                onChange={() => setAreBilant('nu')}
              />
              <span className="ml-2">Nu</span>
            </label>
          </div>
        </div>
      </div>
      
      <div className="space-y-4 mb-4">
        {/* Filtru pentru profit */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Profit/Pierdere (RON)
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Minim</label>
              <input
                type="number"
                value={profitRange.min}
                onChange={(e) => setProfitRange({ ...profitRange, min: e.target.value })}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="Minim"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Maxim</label>
              <input
                type="number"
                value={profitRange.max}
                onChange={(e) => setProfitRange({ ...profitRange, max: e.target.value })}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="Maxim"
              />
            </div>
          </div>
        </div>
        
        {/* Filtru pentru contribuții */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Contribuții la buget (RON)
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Minim</label>
              <input
                type="number"
                value={contributiiRange.min}
                onChange={(e) => setContributiiRange({ ...contributiiRange, min: e.target.value })}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="Minim"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Maxim</label>
              <input
                type="number"
                value={contributiiRange.max}
                onChange={(e) => setContributiiRange({ ...contributiiRange, max: e.target.value })}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="Maxim"
              />
            </div>
          </div>
        </div>
        
        {/* Filtru pentru număr angajați */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Număr angajați
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Minim</label>
              <input
                type="number"
                value={angajatiRange.min}
                onChange={(e) => setAngajatiRange({ ...angajatiRange, min: e.target.value })}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="Minim"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Maxim</label>
              <input
                type="number"
                value={angajatiRange.max}
                onChange={(e) => setAngajatiRange({ ...angajatiRange, max: e.target.value })}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="Maxim"
              />
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Sortare */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Sortare după
          </label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
          >
            <option value="">Implicit</option>
            <option value="PROFIT_PIERDERE">Profit/Pierdere</option>
            <option value="INCASARI LA BUGETUL GENESAL CONSOLIDAT">Contribuții buget</option>
            <option value="NR_MEDIU SALARIATI">Număr angajați</option>
            <option value="FOND_SALARII">Fond salarii</option>
          </select>
        </div>
        
        {/* Direcție sortare */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Direcție sortare
          </label>
          <div className="flex space-x-4 mt-2">
            <label className="inline-flex items-center">
              <input
                type="radio"
                className="form-radio text-blue-600"
                name="sortDirection"
                value="desc"
                checked={sortDirection === 'desc'}
                onChange={() => setSortDirection('desc')}
              />
              <span className="ml-2">Descrescător</span>
            </label>
            <label className="inline-flex items-center">
              <input
                type="radio"
                className="form-radio text-blue-600"
                name="sortDirection"
                value="asc"
                checked={sortDirection === 'asc'}
                onChange={() => setSortDirection('asc')}
              />
              <span className="ml-2">Crescător</span>
            </label>
          </div>
        </div>
      </div>
      
      <div className="flex justify-end space-x-3">
        <button
          type="button"
          onClick={handleResetFilters}
          className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          Resetare
        </button>
        <button
          type="button"
          onClick={handleApplyFilters}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          Aplică filtre
        </button>
      </div>
    </div>
  );
};

export default FinancialFilters;