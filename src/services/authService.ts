// src/services/authService.ts
import { initializeApp } from 'firebase/app';
import { 
  getAuth, 
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  User
} from 'firebase/auth';

// Configurare Firebase
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID
};

// Inițializare Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export interface AuthUser {
  uid: string;
  email: string | null;
  displayName: string | null;
}

// Serviciul de autentificare
export const AuthService = {
  // Înregistrare utilizator nou
  async registerUser(email: string, password: string): Promise<AuthUser> {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      return {
        uid: userCredential.user.uid,
        email: userCredential.user.email,
        displayName: userCredential.user.displayName
      };
    } catch (error: any) {
      console.error('Eroare la înregistrare:', error);
      throw new Error(error.message);
    }
  },
  
  // Autentificare utilizator
  async loginUser(email: string, password: string): Promise<AuthUser> {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      return {
        uid: userCredential.user.uid,
        email: userCredential.user.email,
        displayName: userCredential.user.displayName
      };
    } catch (error: any) {
      console.error('Eroare la autentificare:', error);
      throw new Error(error.message);
    }
  },
  
  // Deconectare utilizator
  async logoutUser(): Promise<void> {
    try {
      await signOut(auth);
    } catch (error: any) {
      console.error('Eroare la deconectare:', error);
      throw new Error(error.message);
    }
  },
  
  // Observer pentru schimbări de autentificare
  onAuthStateChanged(callback: (user: AuthUser | null) => void): () => void {
    return onAuthStateChanged(auth, (user: User | null) => {
      if (user) {
        callback({
          uid: user.uid,
          email: user.email,
          displayName: user.displayName
        });
      } else {
        callback(null);
      }
    });
  },
  
  // Verifică dacă utilizatorul este autentificat
  getCurrentUser(): AuthUser | null {
    const user = auth.currentUser;
    if (!user) return null;
    
    return {
      uid: user.uid,
      email: user.email,
      displayName: user.displayName
    };
  }
};

export default AuthService;