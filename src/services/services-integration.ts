// src/services/dataIntegrationService.ts
import { CasaSchimbType, PunctSchimbType, DateFinanciareType, CountyDistribution, FinancialMetrics } from '../types';
import { normalizeLocationName } from '../utils/normalizationUtils';

/**
 * Serviciu pentru integrarea datelor dintre cele două surse principale:
 * - Date despre case de schimb și puncte de schimb
 * - Date financiare
 */
export const DataIntegrationService = {
  
  /**
   * Asociază datele financiare cu casele de schimb folosind CUI
   */
  associateFinancialData(
    case: CasaSchimbType[], 
    dateFinanciare: DateFinanciareType[], 
    an?: number
  ): (CasaSchimbType & { dateFinanciare?: DateFinanciareType })[] {
    if (!case.length || !dateFinanciare.length) return case;
    
    // Determină anul cel mai recent disponibil în datele financiare dacă nu e specificat
    const targetYear = an || Math.max(...dateFinanciare.map(df => df["AN DE REFERINTA"]));
    
    // Filtrează datele financiare pentru anul țintă
    const relevantFinancialData = dateFinanciare.filter(
      df => df["AN DE REFERINTA"] === targetYear
    );
    
    // Creează un map pentru căutare rapidă după CUI
    const financialDataMap = new Map<number, DateFinanciareType>();
    relevantFinancialData.forEach(data => {
      financialDataMap.set(data["CUI"], data);
    });
    
    // Asociază datele financiare cu casele de schimb
    return case.map(casa => {
      const finData = financialDataMap.get(casa["COD FISCAL CASA"]);
      return {
        ...casa,
        dateFinanciare: finData
      };
    });
  },
  
  /**
   * Calculează distribuția punctelor de schimb pe județe/sectoare
   */
  calculateCountyDistribution(puncte: PunctSchimbType[]): CountyDistribution[] {
    if (!puncte.length) return [];
    
    const distributionMap = new Map<string, { count: number; casaIds: Set<string> }>();
    
    // Agregă datele pe județe normalizate
    puncte.forEach(punct => {
      if (punct["JUDET / SECTOR PUNCT"]) {
        const judetNormalizat = normalizeLocationName(punct["JUDET / SECTOR PUNCT"]);
        
        if (!distributionMap.has(judetNormalizat)) {
          distributionMap.set(judetNormalizat, { count: 0, casaIds: new Set() });
        }
        
        const current = distributionMap.get(judetNormalizat)!;
        current.count += 1;
        current.casaIds.add(punct["COD CASA  CU S"]);
      }
    });
    
    // Convertește la array și sortează după numărul de puncte
    return Array.from(distributionMap.entries())
      .map(([judet, data]) => ({
        judet,
        count: data.count,
        casaIds: Array.from(data.casaIds)
      }))
      .sort((a, b) => b.count - a.count);
  },
  
  /**
   * Calculează metrici financiare agregat pentru casele de schimb
   */
  calculateFinancialMetrics(dateFinanciare: DateFinanciareType[], an?: number): FinancialMetrics {
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
    
    // Determină anul cel mai recent disponibil dacă nu e specificat
    const targetYear = an || Math.max(...dateFinanciare.map(df => df["AN DE REFERINTA"]));
    
    // Filtrează datele pentru anul țintă
    const relevantData = dateFinanciare.filter(
      df => df["AN DE REFERINTA"] === targetYear && df["ARE_BILANT_DA/NU"] === "DA"
    );
    
    let profitTotal = 0;
    let contributiiBuget = 0;
    let numarTotalAngajati = 0;
    let numarCaseProfitabile = 0;
    let numarCaseNeprofitabile = 0;
    let fondSalariiTotal = 0;
    
    // Calculează metricile
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
  },
  
  /**
   * Identifică trenduril pentru o casă de schimb în mai mulți ani
   */
  identifyTrends(cui: number, dateFinanciare: DateFinanciareType[]): any {
    // Filtrare date pentru CUI-ul specificat
    const companyData = dateFinanciare.filter(df => df["CUI"] === cui);
    
    if (companyData.length <= 1) {
      return { hasTrendData: false };
    }
    
    // Sortare după an
    companyData.sort((a, b) => a["AN DE REFERINTA"] - b["AN DE REFERINTA"]);
    
    // Extrage serii de date pentru diferite metrici
    const years = companyData.map(d => d["AN DE REFERINTA"]);
    const profits = companyData.map(d => {
      const profit = typeof d["PROFIT_PIERDERE"] === 'string'
        ? parseFloat(d["PROFIT_PIERDERE"])
        : d["PROFIT_PIERDERE"] || 0;
      return profit;
    });
    
    const budgetContributions = companyData.map(d => {
      const contrib = typeof d["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] === 'string'
        ? parseFloat(d["INCASARI LA BUGETUL GENESAL CONSOLIDAT"])
        : d["INCASARI LA BUGETUL GENESAL CONSOLIDAT"] || 0;
      return contrib;
    });
    
    const employeeCounts = companyData.map(d => {
      const count = typeof d["NR_MEDIU SALARIATI"] === 'string'
        ? parseInt(d["NR_MEDIU SALARIATI"], 10)
        : d["NR_MEDIU SALARIATI"] || 0;
      return count;
    });
    
    // Calculează modificările procentuale
    const profitChange = profits.length > 1 
      ? ((profits[profits.length - 1] - profits[0]) / Math.abs(profits[0])) * 100 
      : 0;
    
    const budgetChange = budgetContributions.length > 1 
      ? ((budgetContributions[budgetContributions.length - 1] - budgetContributions[0]) / budgetContributions[0]) * 100 
      : 0;
    
    const employeeChange = employeeCounts.length > 1 
      ? ((employeeCounts[employeeCounts.length - 1] - employeeCounts[0]) / employeeCounts[0]) * 100 
      : 0;
    
    return {
      hasTrendData: true,
      years,
      profits,
      budgetContributions,
      employeeCounts,
      profitChange,
      budgetChange,
      employeeChange,
      isGrowing: profitChange > 0,
      isIncreasingContribution: budgetChange > 0,
      isHiring: employeeChange > 0
    };
  }
};

export default DataIntegrationService;
