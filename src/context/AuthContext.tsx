// src/context/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AuthService, { AuthUser } from '../services/authService';

interface AuthContextType {
  currentUser: AuthUser | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth trebuie utilizat în interiorul unui AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const unsubscribe = AuthService.onAuthStateChanged((user) => {
      setCurrentUser(user);
      setLoading(false);
    });
    
    return unsubscribe;
  }, []);
  
  const login = async (email: string, password: string): Promise<void> => {
    try {
      await AuthService.loginUser(email, password);
    } catch (error) {
      console.error('Eroare la autentificare:', error);
      throw error;
    }
  };
  
  const register = async (email: string, password: string): Promise<void> => {
    try {
      await AuthService.registerUser(email, password);
    } catch (error) {
      console.error('Eroare la înregistrare:', error);
      throw error;
    }
  };
  
  const logout = async (): Promise<void> => {
    try {
      await AuthService.logoutUser();
    } catch (error) {
      console.error('Eroare la deconectare:', error);
      throw error;
    }
  };
  
  const value = {
    currentUser,
    loading,
    login,
    register,
    logout
  };
  
  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};