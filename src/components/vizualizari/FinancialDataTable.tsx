// src/components/vizualizari/FinancialDataTable.tsx
import React, { useState } from 'react';
import { useExchangeData } from '../../hooks/useExchangeData';
import FinancialFilters from '../filtre/FinancialFilters';

const FinancialDataTable: React.FC = () => {
  const { sortedDateFinanciare, updateFilters } = useExchangeData();
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  
  // Calculăm numărul total de pagini
  const totalPages = Math.ceil(sortedDateFinanciare.length / itemsPerPage);
  
  // Obținem elementele pentru pagina curentă
  const currentItems = sortedDateFinanciare.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );
  
  // Handler pentru schimbarea paginii
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };
  
  // Formatare valori monetare
  const formatCurrency = (value: any): string => {
    if (value === undefined || value === null || value === '') return '-';
    
    const numericValue = typeof value === 'string' ? parseFloat(value) : value;
    
    return numericValue.toLocaleString('ro-RO', {
      maximumFractionDigits: 0
    }) + ' RON';
  };
  
  return (
    <div>
      <div className="mb-6">
        <FinancialFilters 
          onApplyFilters={updateFilters}
        />
      </div>
      
      <div className="bg-white shadow-md rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Denumire
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  An Referință
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Profit/Pierdere
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Contribuții Buget
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Nr. Angajați
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Fond Salarii
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {currentItems.map((item, index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {item["DENUMIRE"]}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item["AN DE REFERINTA"]}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                    parseFloat(item["PROFIT_PIERDERE"] || '0') >= 0 
                      ? 'text-green-600' 
                      : 'text-red-600'
                  }`}>
                    {formatCurrency(item["PROFIT_PIERDERE"])}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatCurrency(item["INCASARI LA BUGETUL GENESAL CONSOLIDAT"])}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item["NR_MEDIU SALARIATI"] || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatCurrency(item["FOND_SALARII"])}
                  </td>
                </tr>
              ))}
              
              {currentItems.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-6 py-4 text-center text-sm text-gray-500">
                    Nu s-au găsit rezultate care să corespundă filtrelor aplicate.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        
        {/* Paginare */}
        {totalPages > 1 && (
          <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
            <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-gray-700">
                  Afișare <span className="font-medium">{(currentPage - 1) * itemsPerPage + 1}</span> la{' '}
                  <span className="font-medium">
                    {Math.min(currentPage * itemsPerPage, sortedDateFinanciare.length)}
                  </span>{' '}
                  din <span className="font-medium">{sortedDateFinanciare.length}</span> rezultate
                </p>
              </div>
              <div>
                <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                  <button
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <span className="sr-only">Anterior</span>
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </button>
                  
                  {/* Afișăm până la 5 pagini, centrând pagina curentă */}
                  {Array.from({ length: Math.min(5, totalPages) }).map((_, i) => {
                    let pageNum;
                    if (totalPages <= 5) {
                      // Dacă avem 5 sau mai puține pagini, afișăm toate
                      pageNum = i + 1;
                    } else if (currentPage <= 3) {
                      // Dacă suntem la început, afișăm primele 5
                      pageNum = i + 1;
                    } else if (currentPage >= totalPages - 2) {
                      // Dacă suntem la sfârșit, afișăm ultimele 5
                      pageNum = totalPages - 4 + i;
                    } else {
                      // Altfel, centrăm pagina curentă
                      pageNum = currentPage - 2 + i;
                    }
                    
                    return (
                      <button
                        key={pageNum}
                        onClick={() => handlePageChange(pageNum)}
                        className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                          currentPage === pageNum
                            ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
                            : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                        }`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                  
                  <button
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <span className="sr-only">Următor</span>
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                    </svg>
                  </button>
                </nav>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FinancialDataTable;