'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowRightIcon, 
  ShieldCheckIcon, 
  ChartBarIcon, 
  BoltIcon,
  StarIcon,
  CheckCircleIcon,
  PlayIcon,
  ArrowDownIcon
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { useInView } from 'react-intersection-observer';
import CountUp from 'react-countup';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';

// Components
import { HeroSection } from '@/components/sections/HeroSection';
import { FeaturesSection } from '@/components/sections/FeaturesSection';
import { StatsSection } from '@/components/sections/StatsSection';
import { TestimonialsSection } from '@/components/sections/TestimonialsSection';
import { PricingSection } from '@/components/sections/PricingSection';
import { CTASection } from '@/components/sections/CTASection';
import { Footer } from '@/components/layout/Footer';
import { Header } from '@/components/layout/Header';
import { FloatingActionButton } from '@/components/ui/FloatingActionButton';
import { CookieConsent } from '@/components/ui/CookieConsent';
import { NewsletterSignup } from '@/components/ui/NewsletterSignup';

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(true);
  const [showCookieConsent, setShowCookieConsent] = useState(false);
  const router = useRouter();

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    // Check cookie consent
    const cookieConsent = localStorage.getItem('cookieConsent');
    if (!cookieConsent) {
      setShowCookieConsent(true);
    }

    return () => clearTimeout(timer);
  }, []);

  const handleGetStarted = () => {
    toast.success('Redirecting to application...');
    router.push('/application');
  };

  const handleCookieAccept = () => {
    localStorage.setItem('cookieConsent', 'accepted');
    setShowCookieConsent(false);
    toast.success('Cookie preferences saved!');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 to-primary-100">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="w-16 h-16 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-2xl font-bold text-primary-900">CreditLens</h2>
          <p className="text-primary-600 mt-2">Loading your credit assessment platform...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main>
        <HeroSection onGetStarted={handleGetStarted} />
        
        <FeaturesSection />
        
        <StatsSection />
        
        <TestimonialsSection />
        
        <PricingSection />
        
        <CTASection onGetStarted={handleGetStarted} />
      </main>

      <Footer />

      {/* Floating Action Button */}
      <FloatingActionButton onClick={handleGetStarted} />

      {/* Cookie Consent */}
      <AnimatePresence>
        {showCookieConsent && (
          <CookieConsent onAccept={handleCookieAccept} />
        )}
      </AnimatePresence>

      {/* Newsletter Signup Modal */}
      <NewsletterSignup />
    </div>
  );
}
