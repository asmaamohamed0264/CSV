// src/types/index.ts

export interface CasaSchimbType {
  "COD CASA  CU S": string;
  "DENUMIRE CASA": string;
  "COD FISCAL CASA": number;
  "COD CASA FARA S": number;
  "SEDIU SOCIAL": string;
  "DATA SEDINTA AUTORIZARE CASA DE SCHIMB": string;
  "DATA AUTORIZARE CASA": string;
  "VALABILITATE CASA": string;
  "JUDET / SECTOR CASA": string;
  "DATA INCHIDERE CASA": string;
}

export interface PunctSchimbType {
  "COD CASA  CU S": string; // Cheia străină către casa de schimb
  "COD PUNCT": string;
  "ADRESA PUNCT": string;
  "DATA SEDINTA AUTORIZARE PUNCT DE LUCRU": string;
  "DATA INCHIDERE PUNCT": string;
  "DENUMIRE CASA": string;
  "DATA AUTORIZARE PUNCT": string;
  "VALABILITATE PUNCT": string;
  "JUDET / SECTOR PUNCT": string;
  isActive?: boolean; // Calculat în funcție de DATA INCHIDERE PUNCT
}

export interface DateFinanciareType {
  "AN DE REFERINTA": number;
  "CUI": number;
  "DENUMIRE": string;
  "ARE_BILANT_DA/NU": string;
  "NR_MEDIU SALARIATI": number | string;
  "FOND_SALARII": number | string;
  "PROFIT_PIERDERE": number | string;
  "INCASARI LA BUGETUL GENESAL CONSOLIDAT": number | string;
  "INCASARI LA BUGETUL DE STAT": number | string;
  "INCASARI BAS (PENSII)": number | string;
  "INCASARI BASS (SANATATE)": number | string;
  "SOMAJ": string;
}

// Tipuri suplimentare pentru filtre și interfețe utilizator
export interface FilterOptions {
  judet: string;
  status: 'active' | 'inactive' | 'all';
  anFinanciar?: number;
  dataStart?: string;
  dataEnd?: string;
  casaId?: string;
  profitMin?: number;
  profitMax?: number;
  numarAngajatiMin?: number;
  numarAngajatiMax?: number;
}

// Tipuri pentru normalizare și transformări
export interface CountyDistribution {
  judet: string;
  count: number;
  casaIds: string[];
}

export interface FinancialMetrics {
  profitTotal: number;
  contributiiBuget: number;
  numarTotalAngajati: number;
  numarCaseProfitabile: number;
  numarCaseNeprofitabile: number;
  fondSalariiTotal: number;
  contributiiMediiPerAngajat: number;
}
