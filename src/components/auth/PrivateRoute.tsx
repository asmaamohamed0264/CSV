// src/components/auth/PrivateRoute.tsx
import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../../context/AuthContext';

interface PrivateRouteProps {
  children: React.ReactNode;
}

const PrivateRoute: React.FC<PrivateRouteProps> = ({ children }) => {
  const { currentUser, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !currentUser) {
      router.push('/login');
    }
  }, [currentUser, loading, router]);

  // Afișează nimic până când verificăm starea de autentificare
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Dacă utilizatorul nu este autentificat, nu afișăm conținutul
  if (!currentUser) {
    return null;
  }

  // Dacă utilizatorul este autentificat, afișăm conținutul protejat
  return <>{children}</>;
};

export default PrivateRoute;