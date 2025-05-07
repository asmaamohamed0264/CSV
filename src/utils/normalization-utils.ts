// src/utils/normalizationUtils.ts

// Mapare pentru forme alternative ale județelor
const JUDETE_MAPPING: Record<string, string> = {
  // Forme standard
  'alba': 'Alba',
  'arad': 'Arad',
  'arges': 'Argeș',
  'argeș': 'Argeș',
  'bacau': 'Bacău',
  'bacău': 'Bacău',
  'bihor': 'Bihor',
  'bistrita-nasaud': 'Bistrița-Năsăud',
  'bistrita nasaud': 'Bistrița-Năsăud',
  'bistriţa-năsăud': 'Bistrița-Năsăud',
  'bistrița-năsăud': 'Bistrița-Năsăud',
  'botosani': 'Botoșani',
  'botoșani': 'Botoșani',
  'brasov': 'Brașov',
  'brașov': 'Brașov',
  'braila': 'Brăila',
  'brăila': 'Brăila',
  'buzau': 'Buzău',
  'buzău': 'Buzău',
  'caras-severin': 'Caraș-Severin',
  'caraș-severin': 'Caraș-Severin',
  'caras severin': 'Caraș-Severin',
  'calarasi': 'Călărași',
  'călărași': 'Călărași',
  'cluj': 'Cluj',
  'constanta': 'Constanța',
  'constanța': 'Constanța',
  'covasna': 'Covasna',
  'dambovita': 'Dâmbovița',
  'dimbovita': 'Dâmbovița',
  'dâmbovița': 'Dâmbovița',
  'dimbovița': 'Dâmbovița',
  'dolj': 'Dolj',
  'galati': 'Galați',
  'galați': 'Galați',
  'giurgiu': 'Giurgiu',
  'gorj': 'Gorj',
  'harghita': 'Harghita',
  'hunedoara': 'Hunedoara',
  'ialomita': 'Ialomița',
  'ialomița': 'Ialomița',
  'iasi': 'Iași',
  'iași': 'Iași',
  'ilfov': 'Ilfov',
  'maramures': 'Maramureș',
  'maramureș': 'Maramureș',
  'mehedinti': 'Mehedinți',
  'mehedinți': 'Mehedinți',
  'mures': 'Mureș',
  'mureș': 'Mureș',
  'neamt': 'Neamț',
  'neamț': 'Neamț',
  'olt': 'Olt',
  'prahova': 'Prahova',
  'satu mare': 'Satu Mare',
  'satu-mare': 'Satu Mare',
  'salaj': 'Sălaj',
  'sălaj': 'Sălaj',
  'sibiu': 'Sibiu',
  'suceava': 'Suceava',
  'teleorman': 'Teleorman',
  'timis': 'Timiș',
  'timiș': 'Timiș',
  'tulcea': 'Tulcea',
  'vaslui': 'Vaslui',
  'valcea': 'Vâlcea',
  'vilcea': 'Vâlcea',
  'vâlcea': 'Vâlcea',
  'vrancea': 'Vrancea',
  
  // Variante pentru București și sectoare
  'bucuresti': 'București',
  'bucurești': 'București',
  'bucuresti sector 1': 'București Sectorul 1',
  'bucurești sector 1': 'București Sectorul 1',
  'bucuresti sectorul 1': 'București Sectorul 1',
  'bucurești sectorul 1': 'București Sectorul 1',
  'sector 1': 'București Sectorul 1',
  'sectorul 1': 'București Sectorul 1',
  'bucuresti sector 2': 'București Sectorul 2',
  'bucurești sector 2': 'București Sectorul 2',
  'bucuresti sectorul 2': 'București Sectorul 2',
  'bucurești sectorul 2': 'București Sectorul 2',
  'sector 2': 'București Sectorul 2',
  'sectorul 2': 'București Sectorul 2',
  'bucuresti sector 3': 'București Sectorul 3',
  'bucurești sector 3': 'București Sectorul 3',
  'bucuresti sectorul 3': 'București Sectorul 3',
  'bucurești sectorul 3': 'București Sectorul 3',
  'sector 3': 'București Sectorul 3',
  'sectorul 3': 'București Sectorul 3',
  'bucuresti sector 4': 'București Sectorul 4',
  'bucurești sector 4': 'București Sectorul 4',
  'bucuresti sectorul 4': 'București Sectorul 4',
  'bucurești sectorul 4': 'București Sectorul 4',
  'sector 4': 'București Sectorul 4',
  'sectorul 4': 'București Sectorul 4',
  'bucuresti sector 5': 'București Sectorul 5',
  'bucurești sector 5': 'București Sectorul 5',
  'bucuresti sectorul 5': 'București Sectorul 5',
  'bucurești sectorul 5': 'București Sectorul 5',
  'sector 5': 'București Sectorul 5',
  'sectorul 5': 'București Sectorul 5',
  'bucuresti sector 6': 'București Sectorul 6',
  'bucurești sector 6': 'București Sectorul 6',
  'bucuresti sectorul 6': 'București Sectorul 6',
  'bucurești sectorul 6': 'București Sectorul 6',
  'sector 6': 'București Sectorul 6',
  'sectorul 6': 'București Sectorul 6',
  
  // Abrevieri comune
  'buc': 'București',
  'buc.': 'București',
  'b': 'București',
  's1': 'București Sectorul 1',
  's2': 'București Sectorul 2',
  's3': 'București Sectorul 3',
  's4': 'București Sectorul 4',
  's5': 'București Sectorul 5',
  's6': 'București Sectorul 6',
};

// Transformă un text prin eliminarea diacriticelor
export function removeDiacritics(text: string): string {
  return text
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();
}

// Normalizează un județ sau sector pentru a obține forma canonică
export function normalizeLocationName(location: string): string {
  if (!location) return '';
  
  // Transformă textul inițial
  const normalized = location
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ');
  
  // Verifică dacă există în mapare
  if (JUDETE_MAPPING[normalized]) {
    return JUDETE_MAPPING[normalized];
  }
  
  // Verifică după eliminarea diacriticelor
  const withoutDiacritics = removeDiacritics(normalized);
  if (JUDETE_MAPPING[withoutDiacritics]) {
    return JUDETE_MAPPING[withoutDiacritics];
  }
  
  // Implementare de fuzzy matching pentru cazuri limită
  // Folosim o abordare simplă pentru a găsi cea mai apropiată potrivire
  const allKeys = Object.keys(JUDETE_MAPPING);
  
  // Funcție pentru a calcula similaritatea între două string-uri
  // Implementăm o versiune simplificată a algoritmului Levenshtein
  const similarity = (a: string, b: string): number => {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;
    
    const matrix: number[][] = [];
    
    // Inițializare matrice
    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }
    
    // Completare matrice
    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i-1) === a.charAt(j-1)) {
          matrix[i][j] = matrix[i-1][j-1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i-1][j-1] + 1, // substituție
            Math.min(
              matrix[i][j-1] + 1, // inserție
              matrix[i-1][j] + 1  // ștergere
            )
          );
        }
      }
    }
    
    return matrix[b.length][a.length];
  };
  
  // Găsește cea mai apropiată potrivire
  let bestMatch = '';
  let bestScore = Infinity;
  
  for (const key of allKeys) {
    const score = similarity(withoutDiacritics, removeDiacritics(key));
    
    // Considerăm o potrivire doar dacă scorul este suficient de bun
    // Scorul reprezintă numărul de modificări necesare pentru a transforma un string în altul
    if (score < bestScore && score <= 3) { // Maxim 3 modificări pentru a fi considerat o potrivire
      bestScore = score;
      bestMatch = key;
    }
  }
  
  if (bestMatch) {
    return JUDETE_MAPPING[bestMatch];
  }
  
  // Dacă nu am găsit nicio potrivire, returnăm textul original cu prima literă mare
  return location.charAt(0).toUpperCase() + location.slice(1).toLowerCase();
}

// Funcție pentru a grupa date după locație (județ/sector) normalizată
export function groupByNormalizedLocation<T>(
  items: T[],
  locationField: keyof T
): Record<string, T[]> {
  const result: Record<string, T[]> = {};
  
  items.forEach(item => {
    const location = item[locationField] as unknown as string;
    const normalizedLocation = normalizeLocationName(location);
    
    if (!result[normalizedLocation]) {
      result[normalizedLocation] = [];
    }
    
    result[normalizedLocation].push(item);
  });
  
  return result;
}

// Funcție pentru a căuta elemente după județ/sector, insensibil la diacritice
export function searchByLocation<T>(
  items: T[],
  locationField: keyof T,
  searchTerm: string
): T[] {
  if (!searchTerm) return items;
  
  const normalizedSearchTerm = removeDiacritics(searchTerm.toLowerCase());
  
  return items.filter(item => {
    const location = item[locationField] as unknown as string;
    if (!location) return false;
    
    const normalizedLocation = removeDiacritics(location.toLowerCase());
    
    return normalizedLocation.includes(normalizedSearchTerm);
  });
}

// Obține lista completă de județe și sectoare în format normalizat
export function getAllNormalizedLocations(): string[] {
  // Extragem toate valorile unice din mapare
  const uniqueLocations = new Set(Object.values(JUDETE_MAPPING));
  return Array.from(uniqueLocations).sort();
}

// Funcție pentru a verifica dacă o locație este un sector din București
export function isBucharestSector(location: string): boolean {
  const normalized = normalizeLocationName(location);
  return normalized.startsWith('București Sectorul ');
}

// Utilitar pentru extragerea sectorului dintr-o locație București
export function extractSectorNumber(location: string): number | null {
  if (!isBucharestSector(location)) return null;
  
  const normalized = normalizeLocationName(location);
  const match = normalized.match(/Sectorul (\d+)/);
  
  if (match && match[1]) {
    return parseInt(match[1], 10);
  }
  
  return null;
}

export default {
  normalizeLocationName,
  removeDiacritics,
  groupByNormalizedLocation,
  searchByLocation,
  getAllNormalizedLocations,
  isBucharestSector,
  extractSectorNumber
};
