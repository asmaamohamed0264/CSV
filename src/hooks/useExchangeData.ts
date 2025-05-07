// src/hooks/useExchangeData.ts
import { useQuery, useQueryClient } from 'react-query';
import { useState, useCallback, useEffect } from 'react';
import GoogleSheetsService from '../services/googleSheetsService';
import { normalizeLocationName } from '../utils/normalizationUtils';
import { 
  CasaSchimbType, 
  PunctSchimbType, 
  DateFinanciareType, 
  DateAngajatiType 
} from '../types';

// Chei pentru query-uri
const QUERY_KEYS = {
  CASE_SCHIMB: 'caseSchimb',
  PUNCTE_SCHIMB: 'puncteSchimb',
  DATE_FINANCIARE: 'dateFinanciare',
  DATE_ANGAJATI: 'dateAngajati'
};

// Interval pentru polling în milisecunde (5 minute)
const POLLING_INTERVAL = 5 * 60 * 1000;

// Interfață pentru opțiunile de filtrare
export interface FilterOptions {
  judet: string;
  status: 'active' | 'inactive' | 'all';
  anFinanciar?: number;
  dataStart?: string;
  dataEnd?: string;
  casaId?: string;
  // Filtre financiare
  financialFilters?: {
    profitMin?: number;
    profitMax?: number;
    contributiiMin?: number;
    contributiiMax?: number;
    angajatiMin?: number;
    angajatiMax?: number;
    anReferinta?: number;
    areBilant?: boolean;
    sortBy?: string;
    sortDirection?: 'asc' | 'desc';
  };
}

/**
 * Custom hook pentru accesarea și filtrarea datelor despre casele de schimb
 */
export function useExchangeData() {
  const queryClient = useQueryClient();
  const [isPollingEnabled, setIsPollingEnabled] = useState(true);
  
  // State pentru filtre
  const [filters, setFilters] = useState<FilterOptions>({
    judet: '',
    status: 'all', // 'active', 'inactive', 'all'
    dataStart: '',
    dataEnd: '',
    caseId: '' // pentru filtrarea punctelor după casă
  });
  
  // Fetch pentru case de schimb
  const {
    data: caseSchimb = [],
    isLoading: isLoadingCase,
    isError: isErrorCase,
    error: errorCase,
    refetch: refetchCase
  } = useQuery(
    [QUERY_KEYS.CASE_SCHIMB, filters],
    () => GoogleSheetsService.getCaseSchimb(),
    {
      refetchInterval: isPollingEnabled ? POLLING_INTERVAL : false,
      staleTime: 60 * 1000, // Datele sunt considerate "proaspete" pentru 60s
      onSuccess: (data) => {
        console.log('Date case schimb actualizate:', data.length);
      }
    }
  );
  
  // Fetch pentru puncte de schimb
  const {
    data: puncteSchimb = [],
    isLoading: isLoadingPuncte,
    isError: isErrorPuncte,
    error: errorPuncte,
    refetch: refetchPuncte
  } = useQuery(
    [QUERY_KEYS.PUNCTE_SCHIMB, filters],
    () => GoogleSheetsService.getPuncteSchimb(),
    {
      refetchInterval: isPollingEnabled ? POLLING_INTERVAL : false,
      staleTime: 60 * 1000,
      onSuccess: (data) => {
        console.log('Date puncte schimb actualizate:', data.length);
      }
    }
  );
  
  // Fetch pentru date financiare
  const {
    data: dateFinanciare = [],
    isLoading: isLoadingFinanciare,
    isError: isErrorFinanciare,
    error: errorFinanciare,
    refetch: refetchFinanciare
  } = useQuery(
    [QUERY_KEYS.DATE_FINANCIARE, filters],
    () => GoogleSheetsService.getDateFinanciare(),
    {
      refetchInterval: isPollingEnabled ? POLLING_INTERVAL : false,
      staleTime: 60 * 1000,
      onSuccess: (data) => {
        console.log('Date financiare actualizate:', data.length);
      }
    }
  );
  
  // Fetch pentru date angajați
  const {
    data: dateAngajati = [],
    isLoading: isLoadingAngajati,
    isError: isErrorAngajati,
    error: errorAngajati,
    refetch: refetchAngajati
  } = useQuery(
    [QUERY_KEYS.DATE_ANGAJATI, filters],
    () => GoogleSheetsService.getDateAngajati(),
    {
      refetchInterval: isPollingEnabled ? POLLING_INTERVAL : false,
      staleTime: 60 * 1000,
      onSuccess: (data) => {
        console.log('Date angajați actualizate:', data.length);
      }
    }
  );
  
  // Funcție pentru a actualiza filtrele
  const updateFilters = useCallback((newFilters: Partial<FilterOptions>) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters
    }));
  }, []);
  
  // Funcție pentru a reîmprospăta toate datele forțat
  const refreshAllData = useCallback(async () => {
    await Promise.all([
      refetchCase(),
      refetchPuncte(),
      refetchFinanciare(),
      refetchAngajati()
    ]);
  }, [refetchCase, refetchPuncte, refetchFinanciare, refetchAngajati]);
  
  // Toggle pentru polling
  const togglePolling = useCallback(() => {
    setIsPollingEnabled(prev => !prev);
  }, []);
  
  // Funcție pentru a invalida cache-ul și a forța reîncărcarea
  const invalidateCache = useCallback(() => {
    queryClient.invalidateQueries(QUERY_KEYS.CASE_SCHIMB);
    queryClient.invalidateQueries(QUERY_KEYS.PUNCTE_SCHIMB);
    queryClient.invalidateQueries(QUERY_KEYS.DATE_FINANCIARE);
    queryClient.invalidateQueries(QUERY_KEYS.DATE_ANGAJATI);
  }, [queryClient]);
  
  // Configurare pentru notificări de actualizare
  useEffect(() => {
    // Funcție callback pentru când se detectează modificări
    const onDataUpdated = () => {
      console.log('Actualizări detectate în datele Google Sheets');
      invalidateCache();
    };
    
    // Configurare polling pe serviciul Google Sheets
    const stopPolling = GoogleSheetsService.setupPolling(5, onDataUpdated);
    
    // Cleanup la demontarea componentei
    return () => {
      stopPolling();
    };
  }, [invalidateCache]);
  
  // Filtrare date case schimb după criterii
  const filteredCaseSchimb = useCallback(() => {
    if (!caseSchimb.length) return [];
    
    return caseSchimb.filter(casa => {
      // Filtrare după județ/sector
      if (filters.judet && casa.judet) {
        const normalizedJudet = normalizeLocationName(casa.judet);
        const normalizedFilter = normalizeLocationName(filters.judet);
        
        if (normalizedJudet !== normalizedFilter) {
          return false;
        }
      }
      
      // Filtrare după status
      if (filters.status !== 'all') {
        const isActive = casa.status === 'activ' || casa.status === 'active';
        if (filters.status === 'active' && !isActive) return false;
        if (filters.status === 'inactive' && isActive) return false;
      }
      
      // Filtrare după dată
      if (filters.dataStart && casa.dataInfiintare) {
        const dataInfiintare = new Date(casa.dataInfiintare);
        const dataStart = new Date(filters.dataStart);
        
        if (dataInfiintare < dataStart) {
          return false;
        }
      }
      
      if (filters.dataEnd && casa.dataInfiintare) {
        const dataInfiintare = new Date(casa.dataInfiintare);
        const dataEnd = new Date(filters.dataEnd);
        
        if (dataInfiintare > dataEnd) {
          return false;
        }
      }
      
      return true;
    });
  }, [caseSchimb, filters]);
  
  // Filtrare date puncte schimb după criterii
  const filteredPuncteSchimb = useCallback(() => {
    if (!puncteSchimb.length) return [];
    
    return puncteSchimb.filter(punct => {
      // Filtrare după casă
      if (filters.caseId && punct.casaId !== filters.caseId) {
        return false;
      }
      
      // Filtrare după județ/sector
      if (filters.judet && punct.judet) {
        const normalizedJudet = normalizeLocationName(punct.judet);
        const normalizedFilter = normalizeLocationName(filters.judet);
        
        if (normalizedJudet !== normalizedFilter) {
          return false;
        }
      }
      
      // Filtrare după status
      if (filters.status !== 'all') {
        const isActive = punct.status === 'activ' || punct.status === 'active';
        if (filters.status === 'active' && !isActive) return false;
        if (filters.status === 'inactive' && isActive) return false;
      }
      
      return true;
    });
  }, [puncteSchimb, filters]);
  
  // Filtrare date financiare după criterii
  const filteredDateFinanciare = useCallback(() => {
    if (!dateFinanciare.length) return [];
    
    return dateFinanciare.filter(data => {
      const financialFilters = filters.financialFilters || {};
      
      // Filtrare după an de referință
      if (financialFilters.anReferinta && data["AN DE REFERINTA"] !== financialFilters.anReferinta) {
        return false;
      }
      
      // Filtrare după existența bilanțului
      if (financialFilters.areBilant !== undefined) {
        const areBilant = data["ARE_BILANT_DA/NU"] === "DA";
        if (financialFilters.areBilant !== areBilant) {
          return false;
        }
      }
      
      // Filtrare după profit/pierdere
      if (financialFilters.profitMin !== undefined || financialFilters.profitMax !== undefined) {
        const profit = typeof data["PROFIT_PIERDERE"] === 'string'
          ? parseFloat(data["PROFIT_PIERDERE"])
          : data["PROFIT_PIERDERE"] || 0;
        
        if (financialFilters.profitMin !== undefined && profit < financialFilters.profitMin) {
          return false;
        }
        
        if (financialFilters.profitMax !== undefined && profit > financialFilters.profitMax) {
          return false;
        }
      }
      
      // Filtrare după contribuții la buget
      if (financialFilters.contributiiMin !== undefined || financialFilters.contributiiMax !== undefined) {
        const contributii = typeof data["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] === 'string'
          ? parseFloat(data["INCASARI LA BUGETUL GENESAL CONSOLIDAT"])
          : data["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] || 0;
        
        if (financialFilters.contributiiMin !== undefined && contributii < financialFilters.contributiiMin) {
          return false;
        }
        
        if (financialFilters.contributiiMax !== undefined && contributii > financialFilters.contributiiMax) {
          return false;
        }
      }
      
      // Filtrare după număr angajați
      if (financialFilters.angajatiMin !== undefined || financialFilters.angajatiMax !== undefined) {
        const angajati = typeof data["NR_MEDIU SALARIATI"] === 'string'
          ? parseInt(data["NR_MEDIU SALARIATI"], 10)
          : data["NR_MEDIU SALARIATI"] || 0;
        
        if (financialFilters.angajatiMin !== undefined && angajati < financialFilters.angajatiMin) {
          return false;
        }
        
        if (financialFilters.angajatiMax !== undefined && angajati > financialFilters.angajatiMax) {
          return false;
        }
      }
      
      return true;
    });
  }, [dateFinanciare, filters.financialFilters]);
  
  // Sortare date financiare
  const sortedDateFinanciare = useCallback(() => {
    const financialFilters = filters.financialFilters || {};
    const filteredData = filteredDateFinanciare();
    
    if (!financialFilters.sortBy || !filteredData.length) {
      return filteredData;
    }
    
    return [...filteredData].sort((a, b) => {
      let valueA, valueB;
      
      // Extragere valori pentru sortare
      if (financialFilters.sortBy === "PROFIT_PIERDERE") {
        valueA = typeof a["PROFIT_PIERDERE"] === 'string'
          ? parseFloat(a["PROFIT_PIERDERE"])
          : a["PROFIT_PIERDERE"] || 0;
        
        valueB = typeof b["PROFIT_PIERDERE"] === 'string'
          ? parseFloat(b["PROFIT_PIERDERE"])
          : b["PROFIT_PIERDERE"] || 0;
      } else if (financialFilters.sortBy === "INCASARI LA BUGETUL GENESAL CONSOLIDAT") {
        valueA = typeof a["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] === 'string'
          ? parseFloat(a["INCASARI LA BUGETUL GENESAL CONSOLIDAT"])
          : a["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] || 0;
        
        valueB = typeof b["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] === 'string'
          ? parseFloat(b["INCASARI LA BUGETUL GENESAL CONSOLIDAT"])
          : b["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] || 0;
      } else if (financialFilters.sortBy === "NR_MEDIU SALARIATI") {
        valueA = typeof a["NR_MEDIU SALARIATI"] === 'string'
          ? parseInt(a["NR_MEDIU SALARIATI"], 10)
          : a["NR_MEDIU SALARIATI"] || 0;
        
        valueB = typeof b["NR_MEDIU SALARIATI"] === 'string'
          ? parseInt(b["NR_MEDIU SALARIATI"], 10)
          : b["NR_MEDIU SALARIATI"] || 0;
      } else if (financialFilters.sortBy === "FOND_SALARII") {
        valueA = typeof a["FOND_SALARII"] === 'string'
          ? parseFloat(a["FOND_SALARII"])
          : a["FOND_SALARII"] || 0;
        
        valueB = typeof b["FOND_SALARII"] === 'string'
          ? parseFloat(b["FOND_SALARII"])
          : b["FOND_SALARII"] || 0;
      } else {
        // Sortare implicită după denumire
        valueA = a["DENUMIRE"] || '';
        valueB = b["DENUMIRE"] || '';
      }
      
      // Aplicare direcție sortare
      const direction = financialFilters.sortDirection === 'asc' ? 1 : -1;
      
      if (valueA < valueB) return -1 * direction;
      if (valueA > valueB) return 1 * direction;
      return 0;
    });
  }, [filteredDateFinanciare, filters.financialFilters]);
  
  // Funcție pentru a obține distribuția după județ/sector
  const getDistributionByLocation = useCallback(() => {
    const puncteByLocation: Record<string, number> = {};
    
    const filteredPuncte = filteredPuncteSchimb();
    
    filteredPuncte.forEach(punct => {
      if (punct.judet) {
        const normalizedLocation = normalizeLocationName(punct.judet);
        
        if (!puncteByLocation[normalizedLocation]) {
          puncteByLocation[normalizedLocation] = 0;
        }
        
        puncteByLocation[normalizedLocation]++;
      }
    });
    
    return puncteByLocation;
  }, [filteredPuncteSchimb]);
  
  // Funcție pentru a corelaționa date financiare cu case de schimb
  const getFinancialDataByExchange = useCallback(() => {
    if (!caseSchimb.length || !dateFinanciare.length) return [];
    
    const filteredCase = filteredCaseSchimb();
    
    return filteredCase.map(casa => {
      // Găsim datele financiare pentru această casă
      const finanteCasa = dateFinanciare.filter(
        date => date.casaId === casa.id
      );
      
      // Calculăm totaluri
      const totalVenituri = finanteCasa.reduce(
        (sum, item) => sum + (parseFloat(item.venituri) || 0), 
        0
      );
      
      const totalContributii = finanteCasa.reduce(
        (sum, item) => sum + (parseFloat(item.contributiiBSC) || 0),
        0
      );
      
      return {
        ...casa,
        finante: finanteCasa,
        totalVenituri,
        totalContributii
      };
    });
  }, [caseSchimb, dateFinanciare, filteredCaseSchimb]);
  
  // Calculare metrici financiare
  const financialMetrics = useCallback(() => {
    if (!dateFinanciare.length) {
      return {
        profitTotal: 0,
        contributiiBuget: 0,
        numarTotalAngajati: 0,
        numarCaseProfitabile: 0,
        numarCaseNeprofitabile: 0,
        fondSalariiTotal: 0,
        contributiiMediiPerAngajat: 0
      };
    }

    const anReferinta = filters.financialFilters?.anReferinta || 
                        Math.max(...dateFinanciare.map(df => df["AN DE REFERINTA"] || 0));
    
    // Filtrează datele pentru anul de referință
    const relevantData = dateFinanciare.filter(
      df => df["AN DE REFERINTA"] === anReferinta && df["ARE_BILANT_DA/NU"] === "DA"
    );
    
    let profitTotal = 0;
    let contributiiBuget = 0;
    let numarTotalAngajati = 0;
    let numarCaseProfitabile = 0;
    let numarCaseNeprofitabile = 0;
    let fondSalariiTotal = 0;
    
    relevantData.forEach(data => {
      // Profit/pierdere
      const profit = typeof data["PROFIT_PIERDERE"] === 'string' 
        ? parseFloat(data["PROFIT_PIERDERE"]) 
        : data["PROFIT_PIERDERE"] || 0;
      
      profitTotal += profit;
      
      if (profit > 0) {
        numarCaseProfitabile += 1;
      } else if (profit < 0) {
        numarCaseNeprofitabile += 1;
      }
      
      // Contribuții buget
      const contributii = typeof data["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] === 'string'
        ? parseFloat(data["INCASARI LA BUGETUL GENESAL CONSOLIDAT"])
        : data["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] || 0;
      
      contributiiBuget += contributii;
      
      // Angajați
      const angajati = typeof data["NR_MEDIU SALARIATI"] === 'string'
        ? parseInt(data["NR_MEDIU SALARIATI"], 10)
        : data["NR_MEDIU SALARIATI"] || 0;
      
      numarTotalAngajati += angajati;
      
      // Fond salarii
      const fondSalarii = typeof data["FOND_SALARII"] === 'string'
        ? parseFloat(data["FOND_SALARII"])
        : data["FOND_SALARII"] || 0;
      
      fondSalariiTotal += fondSalarii;
    });
    
    // Calculează contribuția medie per angajat
    const contributiiMediiPerAngajat = numarTotalAngajati > 0
      ? contributiiBuget / numarTotalAngajati
      : 0;
    
    return {
      profitTotal,
      contributiiBuget,
      numarTotalAngajati,
      numarCaseProfitabile,
      numarCaseNeprofitabile,
      fondSalariiTotal,
      contributiiMediiPerAngajat
    };
  }, [dateFinanciare, filters.financialFilters]);
  
  // Verificarea statusului general de loading
  const isLoading = isLoadingCase || isLoadingPuncte || 
                    isLoadingFinanciare || isLoadingAngajati;
  
  // Verificarea statusului general de eroare
  const hasError = isErrorCase || isErrorPuncte || 
                   isErrorFinanciare || isErrorAngajati;
  
  // Agregarea erorilor
  const errors = [
    isErrorCase && errorCase,
    isErrorPuncte && errorPuncte,
    isErrorFinanciare && errorFinanciare,
    isErrorAngajati && errorAngajati
  ].filter(Boolean);
  
  return {
    // Date brute
    caseSchimb,
    puncteSchimb,
    dateFinanciare,
    dateAngajati,
    
    // Date filtrate
    filteredCaseSchimb: filteredCaseSchimb(),
    filteredPuncteSchimb: filteredPuncteSchimb(),
    filteredDateFinanciare: filteredDateFinanciare(),
    sortedDateFinanciare: sortedDateFinanciare(),
    
    // Agregări și analize
    distributionByLocation: getDistributionByLocation(),
    financialDataByExchange: getFinancialDataByExchange(),
    financialMetrics: financialMetrics(),
    
    // Status
    isLoading,
    hasError,
    errors,
    
    // Acțiuni
    updateFilters,
    refreshAllData,
    togglePolling,
    isPollingEnabled,
    invalidateCache,
    
    // Filtre curente
    filters
  };
}

export default useExchangeData;