// src/services/googleSheetsService.ts
import axios from 'axios';
import { 
  CasaSchimbType, 
  PunctSchimbType, 
  DateFinanciareType, 
  DateAngajatiType 
} from '../types';

// Configurare pentru API
const API_KEY = process.env.NEXT_PUBLIC_GOOGLE_SHEETS_API_KEY;
const SHEET_ID = '12eokegJ46GM9nF86wtCgHu2wDSn9OS0GY_qBWeIUekI';

// Cache pentru date
interface CacheItem<T> {
  data: T[];
  timestamp: number;
  version: string; // Pentru a gestiona versiuni ale datelor
}

// Cache pentru a stoca datele
const dataCache: {
  caseSchimb?: CacheItem<CasaSchimbType>;
  puncteSchimb?: CacheItem<PunctSchimbType>;
  dateFinanciare?: CacheItem<DateFinanciareType>;
  dateAngajati?: CacheItem<DateAngajatiType>;
} = {};

// Durata valabilitate cache (în milisecunde) - 5 minute
const CACHE_TTL = 5 * 60 * 1000;

// Structură pentru gestionarea versiunilor și diffsync
interface SyncMetadata {
  sheetId: string;
  lastSyncTimestamp: number;
  etag: string;
  rowCount: number;
  columnCount: number;
}

// Cache pentru a stoca metadatele de sincronizare
const syncMetadata: Record<string, SyncMetadata> = {};

// Functie pentru a verifica dacă cache-ul este valid
const isCacheValid = <T>(cache?: CacheItem<T>): boolean => {
  if (!cache) return false;
  
  const now = Date.now();
  return now - cache.timestamp < CACHE_TTL;
};

// Implementăm un mecanism de retry pentru a gestiona erorile API
async function fetchWithRetry<T>(
  fetchFn: () => Promise<T>,
  maxRetries = 3,
  delay = 1000
): Promise<T> {
  let lastError: any;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fetchFn();
    } catch (error: any) {
      lastError = error;
      
      // Verifică dacă eroarea este cauzată de rate limiting (429)
      if (error.response && error.response.status === 429) {
        // Așteptăm mai mult timp pentru rate limiting
        await new Promise(resolve => setTimeout(resolve, delay * 2));
      } else {
        // Pentru alte erori, așteptăm doar intervalul standard
        await new Promise(resolve => setTimeout(resolve, delay));
      }
      
      // Crescăm timpul de așteptare exponențial
      delay *= 2;
    }
  }
  
  throw lastError;
}

// Funcția de fetch pentru o foaie specifică
async function fetchSheetData<T>(sheetName: string, range: string): Promise<T[]> {
  try {
    const response = await axios.get(
      `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}/values/${sheetName}!${range}`,
      {
        params: {
          key: API_KEY,
          valueRenderOption: 'UNFORMATTED_VALUE',
          dateTimeRenderOption: 'FORMATTED_STRING'
        }
      }
    );

    // Extrage headerele (prima linie)
    const headers = response.data.values[0];
    
    // Transformă datele în obiecte
    const data = response.data.values.slice(1).map((row: any[]) => {
      const item: Record<string, any> = {};
      
      // Asociază fiecare valoare cu headerul corespunzător
      headers.forEach((header: string, index: number) => {
        item[header] = row[index];
      });
      
      return item as T;
    });

    return data;
  } catch (error) {
    console.error(`Eroare la încărcarea datelor din sheet-ul ${sheetName}:`, error);
    throw error;
  }
}

// Funcție pentru a obține ultima modificare a sheet-ului
async function getSheetLastModified(sheetName: string): Promise<string> {
  try {
    const response = await axios.get(
      `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}`,
      {
        params: {
          key: API_KEY,
          fields: 'sheets.properties'
        }
      }
    );

    const sheets = response.data.sheets;
    const targetSheet = sheets.find((sheet: any) => 
      sheet.properties.title === sheetName
    );
    
    if (targetSheet) {
      return targetSheet.properties.updated || '';
    }
    
    return '';
  } catch (error) {
    console.error('Eroare la obținerea informațiilor despre ultima modificare:', error);
    return '';
  }
}

// Funcție pentru a obține diferențele între actualizări
function getDiff<T>(oldData: T[], newData: T[], idField: keyof T): {
  added: T[],
  updated: T[],
  removed: T[]
} {
  const oldMap = new Map(oldData.map(item => [item[idField], item]));
  const newMap = new Map(newData.map(item => [item[idField], item]));
  
  const added: T[] = [];
  const updated: T[] = [];
  const removed: T[] = [];
  
  // Identifică elementele adăugate și actualizate
  newData.forEach(item => {
    const id = item[idField];
    if (!oldMap.has(id)) {
      added.push(item);
    } else {
      // Verifică dacă a fost actualizat
      const oldItem = oldMap.get(id);
      if (JSON.stringify(oldItem) !== JSON.stringify(item)) {
        updated.push(item);
      }
    }
  });
  
  // Identifică elementele eliminate
  oldData.forEach(item => {
    const id = item[idField];
    if (!newMap.has(id)) {
      removed.push(item);
    }
  });
  
  return { added, updated, removed };
}

// Funcție pentru a obține doar rândurile modificate
async function getModifiedRows<T>(
  sheetName: string, 
  range: string,
  lastSync?: SyncMetadata
): Promise<{data: T[], metadata: SyncMetadata}> {
  try {
    // Obține metadatele curente ale sheet-ului
    const metadataResponse = await axios.get(
      `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}`,
      {
        params: {
          key: API_KEY,
          fields: 'sheets.properties,etag'
        }
      }
    );
    
    const sheetInfo = metadataResponse.data.sheets.find(
      (sheet: any) => sheet.properties.title === sheetName
    );
    
    const currentMetadata: SyncMetadata = {
      sheetId: sheetInfo.properties.sheetId,
      lastSyncTimestamp: Date.now(),
      etag: metadataResponse.data.etag,
      rowCount: sheetInfo.properties.gridProperties.rowCount,
      columnCount: sheetInfo.properties.gridProperties.columnCount
    };
    
    // Verifică dacă au exista