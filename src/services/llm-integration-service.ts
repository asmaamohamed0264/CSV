// src/services/llmService.ts
import axios from 'axios';

// Tipuri pentru LLM
export interface LLMProviderConfig {
  name: string;
  apiEndpoint: string;
  apiKey: string;
  models: string[];
  maxTokens: number;
  streamingSupport: boolean;
  priority: number; // Prioritate pentru selecția automată (mai mic = prioritate mai mare)
}

export interface LLMRequest {
  prompt: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  provider?: string; // Pentru a specifica un anumit provider
  streaming?: boolean;
  onStreamingUpdate?: (text: string) => void;
}

export interface LLMResponse {
  text: string;
  provider: string;
  model: string;
  tokenUsage?: {
    prompt: number;
    completion: number;
    total: number;
  };
}

// Configurări pentru diferite API-uri de LLM
const LLM_PROVIDERS: Record<string, LLMProviderConfig> = {
  openai: {
    name: 'OpenAI',
    apiEndpoint: 'https://api.openai.com/v1/chat/completions',
    apiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY || '',
    models: ['gpt-4-turbo', 'gpt-3.5-turbo'],
    maxTokens: 4096,
    streamingSupport: true,
    priority: 1
  },
  claude: {
    name: 'Claude',
    apiEndpoint: 'https://api.anthropic.com/v1/messages',
    apiKey: process.env.NEXT_PUBLIC_ANTHROPIC_API_KEY || '',
    models: ['claude-3-5-sonnet'],
    maxTokens: 4096,
    streamingSupport: true,
    priority: 2
  },
  deepseek: {
    name: 'DeepSeek',
    apiEndpoint: 'https://api.deepseek.com/v1/chat/completions',
    apiKey: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY || '',
    models: ['deepseek-chat'],
    maxTokens: 2048,
    streamingSupport: false,
    priority: 3
  }
};

// Cache pentru a stoca răspunsuri frecvente
interface CacheItem {
  response: LLMResponse;
  timestamp: number;
}

const responseCache: Record<string, CacheItem> = {};
const CACHE_DURATION = 30 * 60 * 1000; // 30 minute

// Funcție pentru a crea un hash pentru caching
function createCacheKey(request: LLMRequest): string {
  const { prompt, model, maxTokens, temperature, provider } = request;
  return `${provider || 'auto'}-${model || 'default'}-${maxTokens || 0}-${temperature || 0.7}-${prompt}`;
}

// Funcție pentru a determina cel mai bun provider în funcție de disponibilitate și prioritate
function getBestProvider(requestedProvider?: string): LLMProviderConfig | null {
  // Dacă s-a specificat un provider, îl folosim pe acela
  if (requestedProvider && LLM_PROVIDERS[requestedProvider]) {
    const provider = LLM_PROVIDERS[requestedProvider];
    
    // Verificăm dacă are cheia API configurată
    if (provider.apiKey) {
      return provider;
    }
  }
  
  // Altfel, folosim primul provider disponibil în ordinea priorității
  const availableProviders = Object.values(LLM_PROVIDERS)
    .filter(provider => provider.apiKey)
    .sort((a, b) => a.priority - b.priority);
  
  return availableProviders.length > 0 ? availableProviders[0] : null;
}

// Funcție pentru a formata promptul pentru diferite API-uri
function formatPromptForProvider(
  provider: LLMProviderConfig, 
  prompt: string, 
  model: string
): any {
  switch (provider.name) {
    case 'OpenAI':
      return {
        model: model || provider.models[0],
        messages: [
          {
            role: 'system',
            content: 'Ești un asistent specializat în analiza datelor despre case de schimb valutar din România. Răspunde concis și direct, oferind informații relevante și factuale.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 1000,
        temperature: 0.7
      };
      
    case 'Claude':
      return {
        model: model || provider.models[0],
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 1000,
        temperature: 0.7,
        system: 'Ești un asistent specializat în analiza datelor despre case de schimb valutar din România. Răspunde concis și direct, oferind informații relevante și factuale.'
      };
      
    case 'DeepSeek':
      return {
        model: model || provider.models[0],
        messages: [
          {
            role: 'system',
            content: 'Ești un asistent specializat în analiza datelor despre case de schimb valutar din România. Răspunde concis și direct, oferind informații relevante și factuale.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 1000,
        temperature: 0.7
      };
      
    default:
      throw new Error(`Provider necunoscut: ${provider.name}`);
  }
}

// Funcție pentru extragerea răspunsului din diferite formate API
function extractResponseFromProvider(
  provider: LLMProviderConfig,
  responseData: any
): string {
  switch (provider.name) {
    case 'OpenAI':
      return responseData.choices[0]?.message?.content || '';
      
    case 'Claude':
      return responseData.content[0]?.text || '';
      
    case 'DeepSeek':
      return responseData.choices[0]?.message?.content || '';
      
    default:
      throw new Error(`Provider necunoscut: ${provider.name}`);
  }
}

// Funcție pentru a extrage informațiile despre utilizarea token-urilor
function extractTokenUsage(
  provider: LLMProviderConfig,
  responseData: any
): { prompt: number; completion: number; total: number } | undefined {
  try {
    switch (provider.name) {
      case 'OpenAI':
        return {
          prompt: responseData.usage?.prompt_tokens || 0,
          completion: responseData.usage?.completion_tokens || 0,
          total: responseData.usage?.total_tokens || 0
        };
        
      case 'Claude':
        return {
          prompt: responseData.usage?.input_tokens || 0,
          completion: responseData.usage?.output_tokens || 0,
          total: (responseData.usage?.input_tokens || 0) + (responseData.usage?.output_tokens || 0)
        };
        
      case 'DeepSeek':
        return {
          prompt: responseData.usage?.prompt_tokens || 0,
          completion: responseData.usage?.completion_tokens || 0,
          total: responseData.usage?.total_tokens || 0
        };
        
      default:
        return undefined;
    }
  } catch (error) {
    console.error('Eroare la extragerea informațiilor despre utilizarea token-urilor:', error);
    return undefined;
  }
}

// Funcții principale pentru serviciul LLM
export const LLMService = {
  // Obține lista de provideri disponibili
  getAvailableProviders(): LLMProviderConfig[] {
    return Object.values(LLM_PROVIDERS)
      .filter(provider => provider.apiKey)
      .sort((a, b) => a.priority - b.priority);
  },
  
  // Trimite o cerere către un LLM
  async sendRequest(request: LLMRequest): Promise<LLMResponse> {
    const { 
      prompt, 
      model, 
      maxTokens, 
      temperature = 0.7, 
      provider: requestedProvider,
      streaming = false,
      onStreamingUpdate
    } = request;
    
    // Verifică dacă avem răspunsul în cache (doar pentru cereri non-streaming)
    if (!streaming) {
      const cacheKey = createCacheKey(request);
      const cachedResponse = responseCache[cacheKey];
      
      if (cachedResponse && Date.now() - cachedResponse.timestamp < CACHE_DURATION) {
        console.log('Răspuns LLM găsit în cache');
        return cachedResponse.response;
      }
    }
    
    // Determină cel mai bun provider
    const provider = getBestProvider(requestedProvider);
    
    if (!provider) {
      throw new Error('Nu a fost găsit niciun provider LLM disponibil. Verificați cheile API.');
    }
    
    // Folosește modelul specificat sau primul model disponibil
    const selectedModel = model || provider.models[0];
    
    try {
      // Formatează cererea pentru provider-ul selectat
      const requestData = formatPromptForProvider(provider, prompt, selectedModel);
      
      // Adaugă opțiuni pentru streaming dacă este necesar
      if (streaming && provider.streamingSupport && onStreamingUpdate) {
        // Implementare pentru streaming - depinde de fiecare API
        // Aici ar trebui să implementezi logica specifică pentru fiecare provider
        
        // Exemplu simplu (acest cod trebuie adaptat pentru fiecare API)
        const response = await axios.post(
          provider.apiEndpoint,
          {
            ...requestData,
            stream: true
          },
          {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${provider.apiKey}`
            },
            responseType: 'stream'
          }
        );
        
        let accumulatedText = '';
        
        response.data.on('data', (chunk: Buffer) => {
          try {
            const lines = chunk.toString().split('\n').filter(line => line.trim() !== '');
            
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.substring(6);
                
                if (data === '[DONE]') {
                  return;
                }
                
                try {
                  const parsedData = JSON.parse(data);
                  const content = extractResponseFromProvider(provider, parsedData);
                  
                  if (content) {
                    accumulatedText += content;
                    onStreamingUpdate(accumulatedText);
                  }
                } catch (e) {
                  console.error('Eroare la parsarea datelor de streaming:', e);
                }
              }
            }
          } catch (error) {
            console.error('Eroare la procesarea chunk-ului de streaming:', error);
          }
        });
        
        // Returnăm un răspuns placeholder pentru streaming
        // Răspunsul final va fi construit din chunck-urile primite
        return {
          text: accumulatedText,
          provider: provider.name,
          model: selectedModel
        };
      } else {
        // Cerere non-streaming standard
        const response = await axios.post(
          provider.apiEndpoint,
          requestData,
          {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${provider.apiKey}`
            }
          }
        );
        
        // Extrage textul răspunsului
        const responseText = extractResponseFromProvider(provider, response.data);
        
        // Extrage informațiile despre utilizarea token-urilor
        const tokenUsage = extractTokenUsage(provider, response.data);
        
        // Construiește răspunsul
        const llmResponse: LLMResponse = {
          text: responseText,
          provider: provider.name,
          model: selectedModel,
          tokenUsage
        };
        
        // Cachează răspunsul
        const cacheKey = createCacheKey(request);
        responseCache[cacheKey] = {
          response: llmResponse,
          timestamp: Date.now()
        };
        
        return llmResponse;
      }
    } catch (error: any) {
      console.error('Eroare la trimiterea cererii către API-ul LLM:', error);
      
      // Încearcă cu un alt provider dacă primul eșuează
      if (requestedProvider) {
        console.log('Încercare cu un alt provider...');
        return this.sendRequest({
          ...request,
          provider: undefined // Resetează provider-ul pentru a folosi următorul disponibil
        });
      }
      
      throw new Error(`Eroare la comunicarea cu API-ul LLM: ${error.message}`);
    }
  },
  
  // Generează un raport narativ din date
  async generateReport(
    dataContext: any, 
    promptTemplate: string,
    options: Partial<LLMRequest> = {}
  ): Promise<string> {
    // Construiește promptul cu contextul datelor
    const dataContextStr = JSON.stringify(dataContext);
    
    const prompt = `
${promptTemplate}

Date de analizat:
${dataContextStr}

Formatează răspunsul ca un raport scurt, clar și informativ.
`;
    
    // Trimite cererea către LLM
    const response = await this.sendRequest({
      prompt,
      ...options
    });
    
    return response.text;
  },
  
  // Interpretează o întrebare în limbaj natural despre date
  async interpretQuestion(
    question: string,
    dataContext: any,
    options: Partial<LLMRequest> = {}
  ): Promise<string> {
    // Construiește promptul pentru LLM
    const dataContextStr = JSON.stringify(dataContext);
    
    const prompt = `
Întrebarea utilizatorului: "${question}"

Date disponibile pentru a răspunde la întrebare:
${dataContextStr}

Răspunde la întrebarea utilizatorului în mod direct și concis, bazat pe datele furnizate.
`;
    
    // Trimite cererea către LLM
    const response = await this.sendRequest({
      prompt,
      ...options
    });
    
    return response.text;
  },
  
  // Identifică și sugerează insights în date
  async suggestInsights(
    dataContext: any,
    options: Partial<LLMRequest> = {}
  ): Promise<string[]> {
    // Construiește promptul pentru LLM
    const dataContextStr = JSON.stringify(dataContext);
    
    const prompt = `
Analizează următoarele date despre casele de schimb valutar și identifică 3-5 insights valoroase sau tendințe interesante:

${dataContextStr}

Răspunde cu o listă de insights separate prin newline, fără numerotare sau marcatori. Fiecare insight trebuie să fie o propoziție concisă și informativă.
`;
    
    // Trimite cererea către LLM
    const response = await this.sendRequest({
      prompt,
      ...options
    });
    
    // Parsează răspunsul în insights separate
    const insights = response.text
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);
    
    return insights;
  },
  
  // Cache management
  clearCache(): void {
    Object.keys(responseCache).forEach(key => {
      delete responseCache[key];
    });
    
    console.log('Cache LLM golit');
  }
};

export default LLMService;
