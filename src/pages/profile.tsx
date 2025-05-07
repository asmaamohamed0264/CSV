// src/pages/profile.tsx
import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import PrivateRoute from '../components/auth/PrivateRoute';
import { Helmet } from 'react-helmet';

const Profile: React.FC = () => {
  const { currentUser, logout } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  const handleLogout = async () => {
    try {
      setError('');
      setLoading(true);
      await logout();
      // Redirect se face automat prin PrivateRoute
    } catch (error: any) {
      setError('Eroare la deconectare: ' + error.message);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <PrivateRoute>
      <div className="min-h-screen bg-gray-50">
        <Helmet>
          <title>Profil | Case de Schimb Valutar</title>
        </Helmet>
        
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-900">
              Profil Utilizator
            </h1>
          </div>
        </header>
        
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="bg-white shadow rounded-lg p-6">
            {error && (
              <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            )}
            
            {successMessage && (
              <div className="bg-green-50 border-l-4 border-green-400 p-4 mb-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-green-700">{successMessage}</p>
                  </div>
                </div>
              </div>
            )}
            
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium leading-6 text-gray-900">Informații utilizator</h3>
                <div className="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div className="col-span-1">
                    <div className="block text-sm font-medium text-gray-700">Email</div>
                    <div className="mt-1 text-gray-900">{currentUser?.email}</div>
                  </div>
                  <div className="col-span-1">
                    <div className="block text-sm font-medium text-gray-700">UID</div>
                    <div className="mt-1 text-gray-900 text-sm truncate">{currentUser?.uid}</div>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-medium leading-6 text-gray-900">Acțiuni cont</h3>
                <div className="mt-4 flex space-x-4">
                  <button
                    onClick={handleLogout}
                    disabled={loading}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:bg-red-300"
                  >
                    {loading ? 'Se procesează...' : 'Deconectare'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </PrivateRoute>
  );
};

export default Profile;