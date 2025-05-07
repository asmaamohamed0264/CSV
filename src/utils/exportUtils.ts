// src/utils/exportUtils.ts
import * as XLSX from 'xlsx';

interface ExportOptions {
  filename: string;
  sheetName?: string;
  includeHeaders?: boolean;
}

export const ExportUtils = {
  /**
   * Exportă datele în format CSV
   */
  exportToCSV<T>(data: T[], options: ExportOptions): void {
    if (!data.length) {
      console.warn('Nu există date pentru export');
      return;
    }
    
    // Obține headerele din prima înregistrare
    const headers = Object.keys(data[0]);
    
    // Convertește datele în format CSV
    let csvContent = options.includeHeaders !== false ? 
      headers.join(',') + '\n' : '';
    
    // Adaugă rândurile
    csvContent += data.map(row => {
      return headers.map(header => {
        const cell = (row as any)[header];
        // Tratăm virgulele și ghilimelele din celule
        if (cell === null || cell === undefined) {
          return '';
        }
        const cellStr = String(cell);
        if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
          return `"${cellStr.replace(/"/g, '""')}"`;
        }
        return cellStr;
      }).join(',');
    }).join('\n');
    
    // Creează un blob cu datele CSV
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    
    // Descarcă fișierul
    this.downloadFile(blob, options.filename + '.csv');
  },
  
  /**
   * Exportă datele în format Excel
   */
  exportToExcel<T>(data: T[], options: ExportOptions): void {
    if (!data.length) {
      console.warn('Nu există date pentru export');
      return;
    }
    
    // Creează un workbook nou
    const wb = XLSX.utils.book_new();
    
    // Convertește datele în format worksheet
    const ws = XLSX.utils.json_to_sheet(data);
    
    // Adaugă worksheet-ul la workbook
    XLSX.utils.book_append_sheet(
      wb, 
      ws, 
      options.sheetName || 'Sheet1'
    );
    
    // Generează fișierul Excel
    const excelBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
    
    // Creează un blob cu datele Excel
    const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    
    // Descarcă fișierul
    this.downloadFile(blob, options.filename + '.xlsx');
  },
  
  /**
   * Funcție utilitară pentru descărcarea unui fișier
   */
  private downloadFile(blob: Blob, filename: string): void {
    // Creează un URL pentru blob
    const url = window.URL.createObjectURL(blob);
    
    // Creează un element anchor pentru descărcare
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    
    // Adaugă elementul în DOM și declanșează click
    document.body.appendChild(a);
    a.click();
    
    // Curăță
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }
};

export default ExportUtils;