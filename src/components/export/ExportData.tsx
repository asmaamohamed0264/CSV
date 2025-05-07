// src/components/export/ExportData.tsx
import React, { useState } from 'react';
import useExchangeData from '../../hooks/useExchangeData';
import ExportUtils from '../../utils/exportUtils';

interface ExportDataProps {
  className?: string;
}

type ExportFormat = 'csv' | 'excel';
type DataType = 'caseSchimb' | 'puncteSchimb' | 'dateFinanciare' | 'all';

const ExportData: React.FC<ExportDataProps> = ({ className = '' }) => {
  const [isExporting, setIsExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState<ExportFormat>('csv');
  const [dataType, setDataType] = useState<DataType>('all');
  
  const { 
    caseSchimb, 
    puncteSchimb, 
    dateFinanciare,
    filteredCaseSchimb,
    filteredPuncteSchimb
  } = useExchangeData();
  
  const handleExport = async () => {
    setIsExporting(true);
    
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      
      switch (dataType) {
        case 'caseSchimb':
          if (exportFormat === 'csv') {
            ExportUtils.exportToCSV(filteredCaseSchimb, {
              filename: `case-schimb-${timestamp}`
            });
          } else {
            ExportUtils.exportToExcel(filteredCaseSchimb, {
              filename: `case-schimb-${timestamp}`,
              sheetName: 'Case de Schimb'
            });
          }
          break;
          
        case 'puncteSchimb':
          if (exportFormat === 'csv') {
            ExportUtils.exportToCSV(filteredPuncteSchimb, {
              filename: `puncte-schimb-${timestamp}`
            });
          } else {
            ExportUtils.exportToExcel(filteredPuncteSchimb, {
              filename: `puncte-schimb-${timestamp}`,
              sheetName: 'Puncte de Schimb'
            });
          }
          break;
          
        case 'dateFinanciare':
          if (exportFormat === 'csv') {
            ExportUtils.exportToCSV(dateFinanciare, {
              filename: `date-financiare-${timestamp}`
            });
          } else {
            ExportUtils.exportToExcel(dateFinanciare, {
              filename: `date-financiare-${timestamp}`,
              sheetName: 'Date Financiare'
            });
          }
          break;
          
        case 'all':
          if (exportFormat === 'excel') {
            // Pentru Excel putem pune toate datele în fișiere separate
            const wb = XLSX.utils.book_new();
            
            const wsCaseSchimb = XLSX.utils.json_to_sheet(filteredCaseSchimb);
            XLSX.utils.book_append_sheet(wb, wsCaseSchimb, 'Case de Schimb');
            
            const wsPuncteSchimb = XLSX.utils.json_to_sheet(filteredPuncteSchimb);
            XLSX.utils.book_append_sheet(wb, wsPuncteSchimb, 'Puncte de Schimb');
            
            const wsDateFinanciare = XLSX.utils.json_to_sheet(dateFinanciare);
            XLSX.utils.book_append_sheet(wb, wsDateFinanciare, 'Date Financiare');
            
            const excelBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
            const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
            ExportUtils.downloadFile(blob, `date-complete-${timestamp}.xlsx`);
          } else {
            // Pentru CSV facem fișiere separate
            ExportUtils.exportToCSV(filteredCaseSchimb, {
              filename: `case-schimb-${timestamp}`
            });
            ExportUtils.exportToCSV(filteredPuncteSchimb, {
              filename: `puncte-schimb-${timestamp}`
            });
            ExportUtils.exportToCSV(dateFinanciare, {
              filename: `date-financiare-${timestamp}`
            });
          }
          break;
      }
    } catch (error) {
      console.error('Eroare la exportul datelor:', error);
      alert('A apărut o eroare la exportul datelor. Vă rugăm să încercați din nou.');
    } finally {
      setIsExporting(false);
    }
  };
  
  return (
    <div className={`export-data ${className}`}>
      <div className="p-4 bg-white rounded-lg shadow border">
        <h3 className="text-lg font-semibold mb-4">Export Date</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Format Export
            </label>
            <div className="flex space-x-4">
              <label className="inline-flex items-center">
                <input 
                  type="radio" 
                  className="form-radio text-blue-600"
                  name="exportFormat"
                  value="csv"
                  checked={exportFormat === 'csv'}
                  onChange={() => setExportFormat('csv')}
                />
                <span className="ml-2">CSV</span>
              </label>
              <label className="inline-flex items-center">
                <input 
                  type="radio" 
                  className="form-radio text-blue-600"
                  name="exportFormat"
                  value="excel"
                  checked={exportFormat === 'excel'}
                  onChange={() => setExportFormat('excel')}
                />
                <span className="ml-2">Excel</span>
              </label>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Tip Date
            </label>
            <select
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              value={dataType}
              onChange={(e) => setDataType(e.target.value as DataType)}
            >
              <option value="all">Toate datele</option>
              <option value="caseSchimb">Case de schimb</option>
              <option value="puncteSchimb">Puncte de schimb</option>
              <option value="dateFinanciare">Date financiare</option>
            </select>
          </div>
          
          <div>
            <button
              type="button"
              onClick={handleExport}
              disabled={isExporting}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-blue-300"
            >
              {isExporting ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Se exportă...
                </>
              ) : (
                <>
                  <svg className="-ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Export
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExportData;